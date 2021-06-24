//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) VPP application
/// using the core API subset.  For more information see:
/// https://software.intel.com/content/www/us/en/develop/articles/upgrading-from-msdk-to-onevpl.html
/// https://oneapi-src.github.io/oneAPI-spec/elements/oneVPL/source/index.html
///
/// @file

#include <drm/drm_fourcc.h>
#include <level_zero/ze_api.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
#include <va/va_vpp.h>

#include <CL/sycl.hpp>
#include <CL/sycl/backend.hpp>
#include <CL/sycl/backend/level_zero.hpp>

#include "util.h"
#include "vaapi_allocator.h"

#define OUTPUT_WIDTH 256
#define OUTPUT_HEIGHT 192
#define OUTPUT_FILE "out.raw"
#define BLUR_RADIUS 5
#define BLUR_SIZE (float)((BLUR_RADIUS << 1) + 1)
#define USE_LEVEL_ZERO_IPC
#define MAX_PLANES_NUMBER 4

#define USE_BLUR

struct usm_image_context {
  ze_context_handle_t ze_context;
  void *ptr;
  uint64_t drm_format_modifier;

  uint32_t planes_count;

  uint32_t offset[MAX_PLANES_NUMBER];
  uint32_t pitch[MAX_PLANES_NUMBER];
};

#ifdef USE_BLUR
// DPC++ kernel for image blurring
void BlurFrame(sycl::queue q, int width, int height, uint8_t *src_ptr,
               size_t src_stride, uint8_t *dst_ptr, size_t dst_stride) {
  try {
    q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
       auto y = idx.get(0);
       auto x = idx.get(1);

       // Compute average intensity. Skip borders and set to black color
       float t0 = 0, t1 = 0, t2 = 0;

       if (x >= BLUR_RADIUS && x < width - BLUR_RADIUS && y >= BLUR_RADIUS &&
           y < height - BLUR_RADIUS) {
         for (int yy = y - BLUR_RADIUS; yy < y + BLUR_RADIUS; yy++) {
           for (int xx = x - BLUR_RADIUS; xx < x + BLUR_RADIUS; xx++) {
             t0 += src_ptr[yy * src_stride + 4 * xx];
             t1 += src_ptr[yy * src_stride + 4 * xx + 1];
             t2 += src_ptr[yy * src_stride + 4 * xx + 2];
           }
         }
         t0 /= BLUR_SIZE * BLUR_SIZE;
         t1 /= BLUR_SIZE * BLUR_SIZE;
         t2 /= BLUR_SIZE * BLUR_SIZE;
       }

       dst_ptr[y * dst_stride + 4 * x + 0] = t0;
       dst_ptr[y * dst_stride + 4 * x + 1] = t1;
       dst_ptr[y * dst_stride + 4 * x + 2] = t2;
       dst_ptr[y * dst_stride + 4 * x + 3] =
           src_ptr[y * src_stride + 4 * x + 3];
     }).wait_and_throw();
  } catch (std::exception e) {
    std::cout << "  SYCL exception caught: " << e.what() << std::endl;
    return;
  }
}
#else
void simple_pixel_flip(cl::sycl::queue deviceQueue, mfxU8 *VA, mfxU8 *VC,
                       unsigned long N) {
  constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
  constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

  cl::sycl::buffer<mfxU8, 1> bufferA(VA, cl::sycl::range<1>(N));
  cl::sycl::buffer<mfxU8, 1> bufferC(VC, cl::sycl::range<1>(N));

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    auto kern = [=](cl::sycl::id<1> wiID) {
      accessorC[wiID] = 255 - accessorA[wiID];
    };
    cgh.parallel_for(cl::sycl::range<1>(N), kern);
  });
}
#endif

void Usage(void) {
  printf("\n");
  printf("   Usage  :  legacy-vpp\n");
  printf("     -hw        use hardware implementation\n");
  printf("     -sw        use software implementation\n");
  printf("     -i input file name (sw=I420 raw frames, hw=NV12)\n");
  printf("     -w input width\n");
  printf("     -h input height\n\n");
  printf("   Example:  legacy-vpp -i in.i420 -w 128 -h 96\n");
  printf(
      "   To view:  ffplay -f rawvideo -pixel_format yuv420p -video_size %dx%d "
      "-pixel_format yuv420p %s\n\n",
      OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FILE);
  printf(" * Resize raw frames to %dx%d size in %s\n\n", OUTPUT_WIDTH,
         OUTPUT_HEIGHT, OUTPUT_FILE);
#ifdef USE_BLUR
  printf(
      "   Blur VPP output by using DPCPP kernel (default kernel size is "
      "[%d]x[%d]) in %s\n",
      2 * BLUR_RADIUS + 1, 2 * BLUR_RADIUS + 1, OUTPUT_FILE);
#endif
  return;
}

int main(int argc, char **argv) {
  sycl::queue q = sycl::gpu_selector{};

  mfxStatus sts = MFX_ERR_NONE;
  bool isDraining = false;
  bool isStillGoing = true;
  FILE *sink = NULL;
  FILE *source = NULL;
  mfxSession session;
  mfxFrameAllocator mfxAllocator;
  mfxFrameAllocRequest VPPRequest[2] = {};  // [0] - in, [1] - out
  mfxVideoParam VPPParams = {};
  mfxU32 framenum = 0;

  int nIndexVPPInSurf = 0;
  int nIndexVPPOutSurf = 0;
  mfxFrameAllocResponse mfxResponseIn;
  mfxFrameAllocResponse mfxResponseOut;

#ifdef USE_BLUR
  mfxU8 *blur_data = NULL;
  mfxU32 blur_data_size = 0;
  size_t blur_pitch = 0;
#else
  mfxU8 *out_data;
#endif

  mfxFrameSurface1 *vppInSurfacePool = NULL;
  mfxFrameSurface1 *vppOutSurfacePool = NULL;
  Params cliParams = {};

  mfxSyncPoint syncp;

  mfxConfig cfg[1];
  mfxLoader loader = NULL;

  // Parse command line args to cliParams
  if (ParseArgsAndValidate(argc, argv, &cliParams, PARAMS_VPP) == false) {
    Usage();
    return 1;  // return 1 as error code
  }

  source = fopen(cliParams.infileName, "rb");
  VERIFY(source, "Could not open input file");

  sink = fopen(OUTPUT_FILE, "wb");
  VERIFY(sink, "Could not create output file");

#ifdef USE_BLUR
  // Allocate memory for blurred frame
  blur_data_size = GetSurfaceSize(MFX_FOURCC_BGRA, OUTPUT_WIDTH, OUTPUT_HEIGHT);
  blur_pitch = OUTPUT_WIDTH * 4;

  blur_data = sycl::malloc_shared<mfxU8>(blur_data_size, q);
#else
  out_data = sycl::malloc_shared<mfxU8>(
      GetSurfaceSize(MFX_FOURCC_NV12, OUTPUT_WIDTH, OUTPUT_HEIGHT), q);
#endif

  // Initialize oneVPL session
  loader = MFXLoad();
  VERIFY(NULL != loader, "MFXLoad failed -- is implementation in path?");

  // Implementation used must be the type requested from command line
  cfg[0] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[0], "MFXCreateConfig failed")

  sts = MFXSetConfigFilterProperty(cfg[0], (mfxU8 *)"mfxImplDescription.Impl",
                                   cliParams.implValue);
  VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for Impl");

  sts = MFXCreateSession(loader, 0, &session);
  VERIFY(MFX_ERR_NONE == sts,
         "Cannot create session -- no implementations meet selection criteria");

  // Print info about implementation loaded
  ShowImplementationInfo(loader, 0);

  // open VA display, set handle, and set allocator
  va_dpy = (VADisplay)InitAcceleratorHandle(session);

  // set up VAAPI surface allocator
  mfxAllocator.pthis = &session;
  mfxAllocator.Alloc = simple_alloc;
  mfxAllocator.Free = simple_free;
  mfxAllocator.Lock = simple_lock;
  mfxAllocator.Unlock = simple_unlock;
  mfxAllocator.GetHDL = simple_gethdl;

  // Since we are using video memory we must provide Media SDK with an external
  // allocator
  sts = MFXVideoCORE_SetFrameAllocator(session, &mfxAllocator);
  VERIFY(MFX_ERR_NONE == sts, "SetFrameAllocator failed");

  // Input data
  VPPParams.vpp.In.FourCC = MFX_FOURCC_NV12;
  VPPParams.vpp.In.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
  VPPParams.vpp.In.CropW = cliParams.srcWidth;
  VPPParams.vpp.In.CropH = cliParams.srcHeight;
  VPPParams.vpp.In.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
  VPPParams.vpp.In.FrameRateExtN = 30;
  VPPParams.vpp.In.FrameRateExtD = 1;
  VPPParams.vpp.In.Width = ALIGN16(VPPParams.vpp.In.CropW);
  VPPParams.vpp.In.Height = ALIGN16(VPPParams.vpp.In.CropH);

  // Output data
#ifdef USE_BLUR
  VPPParams.vpp.Out.FourCC = MFX_FOURCC_BGRA;
#else
  VPPParams.vpp.Out.FourCC = MFX_FOURCC_NV12;
#endif
  VPPParams.vpp.Out.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
  VPPParams.vpp.Out.CropW = OUTPUT_WIDTH;
  VPPParams.vpp.Out.CropH = OUTPUT_HEIGHT;
  VPPParams.vpp.Out.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
  VPPParams.vpp.Out.FrameRateExtN = 30;
  VPPParams.vpp.Out.FrameRateExtD = 1;
  VPPParams.vpp.Out.Width = ALIGN16(VPPParams.vpp.Out.CropW);
  VPPParams.vpp.Out.Height = ALIGN16(VPPParams.vpp.Out.CropH);

  VPPParams.IOPattern =
      MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;

  // Initialize Media SDK VPP
  sts = MFXVideoVPP_Init(session, &VPPParams);
  VERIFY(MFX_ERR_NONE == sts, "Could not initialize VPP");

  // Query number of required surfaces for VPP
  sts = MFXVideoVPP_QueryIOSurf(session, &VPPParams, VPPRequest);
  VERIFY(MFX_ERR_NONE == sts, "Error in QueryIOSurf");

  // Allocate required surfaces
  sts = mfxAllocator.Alloc(mfxAllocator.pthis, &VPPRequest[0], &mfxResponseIn);
  VERIFY(MFX_ERR_NONE == sts, "Error allocating input surfaces");

  sts = mfxAllocator.Alloc(mfxAllocator.pthis, &VPPRequest[1], &mfxResponseOut);
  VERIFY(MFX_ERR_NONE == sts, "Error allocating output surfaces");

  // Allocate surface headers (mfxFrameSurface1) for VPP
  vppInSurfacePool = (mfxFrameSurface1 *)calloc(sizeof(mfxFrameSurface1),
                                                mfxResponseIn.NumFrameActual);
  for (int i = 0; i < mfxResponseIn.NumFrameActual; i++) {
    memset(&vppInSurfacePool[i], 0, sizeof(mfxFrameSurface1));
    vppInSurfacePool[i].Info = VPPParams.vpp.In;
    vppInSurfacePool[i].Data.MemId =
        mfxResponseIn
            .mids[i];  // MID (memory id) represent one D3D NV12 surface
  }

  vppOutSurfacePool = (mfxFrameSurface1 *)calloc(sizeof(mfxFrameSurface1),
                                                 mfxResponseOut.NumFrameActual);
  for (int i = 0; i < mfxResponseOut.NumFrameActual; i++) {
    memset(&vppOutSurfacePool[i], 0, sizeof(mfxFrameSurface1));
    vppOutSurfacePool[i].Info = VPPParams.vpp.Out;
    vppOutSurfacePool[i].Data.MemId =
        mfxResponseOut
            .mids[i];  // MID (memory id) represent one D3D NV12 surface
  }

  // ===================================
  // Start processing the frames
  //

  printf("Processing %s -> %s\n", cliParams.infileName, OUTPUT_FILE);

  while (isStillGoing == true) {
    if (isDraining == false) {
      nIndexVPPInSurf = GetFreeSurfaceIndex(
          vppInSurfacePool,
          mfxResponseIn.NumFrameActual);  // Find free input frame surface

      sts = mfxAllocator.Lock(mfxAllocator.pthis,
                              vppInSurfacePool[nIndexVPPInSurf].Data.MemId,
                              &(vppInSurfacePool[nIndexVPPInSurf].Data));
      VERIFY(MFX_ERR_NONE == sts, "Error locking surface for ReadRawFrame");

      sts = ReadRawFrame(&vppInSurfacePool[nIndexVPPInSurf],
                         source);  // Load frame from file into surface
      if (sts == MFX_ERR_MORE_DATA)
        isDraining = true;
      else
        VERIFY(MFX_ERR_NONE == sts, "Unknown error reading input");

      sts = mfxAllocator.Unlock(mfxAllocator.pthis,
                                vppInSurfacePool[nIndexVPPInSurf].Data.MemId,
                                &(vppInSurfacePool[nIndexVPPInSurf].Data));
      VERIFY(MFX_ERR_NONE == sts, "Error unlocking surface for ReadRawFrame");
    }

    nIndexVPPOutSurf = GetFreeSurfaceIndex(
        vppOutSurfacePool,
        mfxResponseOut.NumFrameActual);  // Find free output frame surface

    sts = MFXVideoVPP_RunFrameVPPAsync(
        session,
        (isDraining == true) ? NULL : &vppInSurfacePool[nIndexVPPInSurf],
        &vppOutSurfacePool[nIndexVPPOutSurf], NULL, &syncp);

    switch (sts) {
      case MFX_ERR_NONE: {
        sts = MFXVideoCORE_SyncOperation(session, syncp,
                                         WAIT_100_MILLISECONDS * 1000);
        VERIFY(MFX_ERR_NONE == sts, "Error in SyncOperation");

#if 0            
            {
                mfxMemId mid = vppOutSurfacePool[nIndexVPPOutSurf].Data.MemId;
                mfxFrameData *ptr = &(vppOutSurfacePool[nIndexVPPOutSurf].Data);
                vaapiMemId *vaapi_mid = (vaapiMemId *)mid;
                mfxU8 *pBuffer = 0;

                vaSyncSurface(va_dpy, *(vaapi_mid->m_surface));
                vaDeriveImage(va_dpy, *(vaapi_mid->m_surface),&(vaapi_mid->m_image));
                vaMapBuffer(va_dpy, vaapi_mid->m_image.buf, (void **)&pBuffer);
                ptr->Pitch =
                    (mfxU16)vaapi_mid->m_image.pitches[0];
                ptr->Y =
                    pBuffer +
                    vaapi_mid->m_image.offsets[0];
                ptr->U =
                    pBuffer +
                    vaapi_mid->m_image.offsets[1];
                ptr->V = ptr->U + 1;
            }
#endif
        mfxFrameSurface1 *pmfxOutSurface;
        pmfxOutSurface = &vppOutSurfacePool[nIndexVPPOutSurf];

        mfxHDL handle = NULL;
        sts = mfxAllocator.GetHDL(mfxAllocator.pthis, pmfxOutSurface->Data.MemId, &handle);
        VERIFY(MFX_ERR_NONE == sts, "Error in mfxAllocator.GetHDL");
        
        VASurfaceID va_surface_id = *(VASurfaceID*)handle;

        usm_image_context context;
        VADRMPRIMESurfaceDescriptor prime_desc = {};

        vaExportSurfaceHandle(va_dpy, va_surface_id,
                              VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                              VA_EXPORT_SURFACE_READ_ONLY, &prime_desc);
        int dma_fd = prime_desc.objects[0].fd;

        context.drm_format_modifier =
            prime_desc.objects[0]
                .drm_format_modifier;  // non-zero if tiled (non-linear) mem

        uint32_t n_planes = 0;
        for (uint32_t i = 0; i < prime_desc.num_layers; i++) {
          auto layer = &prime_desc.layers[i];
          for (uint32_t j = 0; j < layer->num_planes; j++) {
            if (n_planes < MAX_PLANES_NUMBER) {
              context.pitch[n_planes] = layer->pitch[j];
              context.offset[n_planes] = layer->offset[j];
              n_planes++;
            }
          }
        }
        context.planes_count = n_planes;

        ze_context_handle_t ze_context =
            q.get_context().get_native<sycl::backend::level_zero>();
        context.ze_context = ze_context;
        ze_device_handle_t ze_device =
            q.get_device().get_native<sycl::backend::level_zero>();
        void *ptr = nullptr;
        ze_result_t ze_res;

#ifdef USE_LEVEL_ZERO_IPC
        // TODO: use zeMemAllocDevice call below, once it supported in
        // level-zero
        ze_ipc_mem_handle_t ipc_handle{};
        *(size_t *)ipc_handle.data = dma_fd;
        ze_res = zeMemOpenIpcHandle(ze_context, ze_device, ipc_handle, 0, &ptr);
#else
        // use this version -- it is the preferred approach
        ze_device_mem_alloc_desc_t alloc_desc = {};
        alloc_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
        ze_external_memory_import_fd_t import_fd = {
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
            nullptr,  // pNext
            ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, dma_fd};
        alloc_desc.pNext = &import_fd;
        ze_res =
            zeMemAllocDevice(ze_context, &alloc_desc,
                             prime_desc.objects[0].size, 0, ze_device, &ptr);
#endif

        if (ze_res != ZE_RESULT_SUCCESS) {
          throw std::runtime_error("Failed to get USM pointer");
        }

        close(dma_fd);
        context.ptr = ptr;

        int W = prime_desc.width;
        int H = prime_desc.height;

#ifdef USE_BLUR

        BlurFrame(q, OUTPUT_WIDTH, OUTPUT_HEIGHT, (uint8_t *)ptr,
                  prime_desc.layers[0].pitch[0], blur_data, blur_pitch);

        for (int r = 0; r < OUTPUT_HEIGHT; r++) {
          fwrite(blur_data + (r * blur_pitch), 1, blur_pitch, sink);
        }
#else

        simple_pixel_flip(q, pmfxOutSurface->Data.Y, out_data, W * H * 1.5);

        for (int r = 0; r < H; r++) {
          fwrite(out_data + (r * pmfxOutSurface->Data.Pitch), 1, W, sink);
        }

        for (int r = 0; r < H * .5; r++) {
          fwrite(out_data + prime_desc.layers[1].offset[0] +
                     (r * pmfxOutSurface->Data.Pitch),
                 1, W, sink);
        }
#endif

        // unmap

#ifdef USE_LEVEL_ZERO_IPC
        ze_res = zeMemCloseIpcHandle(ze_context, ptr);
#else
        ze_res = zeMemFree(ze_context, ptr);
#endif

        printf("Frame number: %d\r", ++framenum);
        fflush(stdout);
      } break;
      case MFX_ERR_MORE_DATA:
        // Need more input frames before VPP can produce an output
        if (isDraining) isStillGoing = false;
        break;
      case MFX_ERR_MORE_SURFACE:
        // The output frame is ready after synchronization.
        // Need more surfaces at output for additional output frames available.
        // This applies to external memory allocations and should not be
        // expected for a simple internal allocation case like this
        break;
      case MFX_ERR_DEVICE_LOST:
        // For non-CPU implementations,
        // Cleanup if device is lost
        break;
      case MFX_WRN_DEVICE_BUSY:
        usleep(WAIT_100_MILLISECONDS * 1000);
        // For non-CPU implementations,
        // Wait a few milliseconds then try again
        break;
      default:
        printf("unknown status %d\n", sts);
        isStillGoing = false;
        break;
    }
  }

end:
  printf("processed %d frames\n", framenum);

  mfxAllocator.Free(mfxAllocator.pthis, &mfxResponseIn);
  mfxAllocator.Free(mfxAllocator.pthis, &mfxResponseOut);
  // free kernel data

#ifdef USE_BLUR
  sycl::free(blur_data, q);
#else
  sycl::free(out_data, q);
#endif

  return 0;
}
