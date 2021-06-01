//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) VPP application,
/// using 2.x API with internal memory management
///
/// @file

#include "util.h"

#ifdef BUILD_DPCPP
#include "CL/sycl.hpp"
#define BLUR_RADIUS 5
#define BLUR_SIZE (float)((BLUR_RADIUS << 1) + 1)

#ifdef LIBVA_SUPPORT
//#define USE_LEVEL_ZERO_IPC_WORKAROUND
#include <assert.h>
#include <drm/drm_fourcc.h>
#include <fcntl.h>
#include <level_zero/ze_api.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
#include <va/va_vpp.h>

#include <CL/sycl.hpp>
#include <CL/sycl/backend.hpp>
#include <CL/sycl/backend/level_zero.hpp>

#define MAX_PLANES_NUMBER 4
struct usm_image_context {
  ze_context_handle_t ze_context;
  void *ptr;
  uint64_t drm_format_modifier;

  uint32_t planes_count;

  uint32_t offset[MAX_PLANES_NUMBER];
  uint32_t pitch[MAX_PLANES_NUMBER];
};

struct vaapiMemId {
  VASurfaceID *surf;
};

#endif

#endif

#define OUTPUT_WIDTH 256
#define OUTPUT_HEIGHT 192
#define OUTPUT_FILE "out.raw"
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

void Usage(void) {
  printf("\n");
#ifdef BUILD_DPCPP
  printf(" ! Blur feature enabled by using DPCPP\n\n");
#else
  printf(" ! Blur feature disabled\n\n");
#endif
  printf("   Usage  :  dpcpp-blur\n");
  printf("     -hw        use hardware implementation\n");
  printf("     -sw        use software implementation\n");
  printf("     -i input file name (sw=I420 raw frames, hw=NV12)\n");
  printf("     -w input width\n");
  printf("     -h input height\n\n");
  printf("   Example:  dpcpp-blur -i in.i420 -w 128 -h 96 -sw\n");
  printf(
      "   To view:  ffplay -f rawvideo -pixel_format bgra -video_size %dx%d "
      "%s\n\n",
      OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FILE);

#ifdef BUILD_DPCPP
  printf(
      "   Blur VPP output by using DPCPP kernel (default kernel size is "
      "[%d]x[%d]) in %s\n",
      2 * BLUR_RADIUS + 1, 2 * BLUR_RADIUS + 1, OUTPUT_FILE);
#endif
  return;
}

#ifdef BUILD_DPCPP
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
      accessorC[wiID] = 255;
      if ((wiID % 4) != 3) {
        accessorC[wiID] = 255 - accessorA[wiID];
      }
    };
    cgh.parallel_for(cl::sycl::range<1>(N), kern);
  });
}
#endif

int main(int argc, char *argv[]) {
  // Variables used for legacy and 2.x
  bool isDraining = false;
  bool isStillGoing = true;
  FILE *sink = NULL;
  FILE *source = NULL;
  mfxFrameSurface1 *vppInSurface = NULL;
  mfxFrameSurface1 *vppOutSurface = NULL;
  mfxSession session = NULL;
  mfxSyncPoint syncp = {};
  mfxU32 framenum = 0;
  mfxStatus sts = MFX_ERR_NONE;
  mfxStatus sts_r = MFX_ERR_NONE;
  Params cliParams = {};
  void *accelHandle = NULL;
  mfxVideoParam VPPParams = {};
  mfxU8 *blur_data = nullptr;

  // variables used only in 2.x version
  mfxConfig cfg[3];
  mfxVariant cfgVal[3];
  mfxLoader loader = NULL;

  // Parse command line args to cliParams
  if (ParseArgsAndValidate(argc, argv, &cliParams, PARAMS_VPP) == false) {
    Usage();
    return 1;  // return 1 as error code
  }

#ifdef BUILD_DPCPP
  printf("\n! DPCPP blur feature enabled\n\n");

  // Create SYCL execution queue
  sycl::queue q = (MFX_IMPL_SOFTWARE == cliParams.impl)
                      ? sycl::queue(sycl::cpu_selector())
                      : sycl::queue(sycl::gpu_selector());

  // Print device name selected for this queue.
  std::cout << "  Queue initialized on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl
            << std::endl;
#else
  printf("\n! DPCPP blur feature not enabled\n\n");
#endif

  source = fopen(cliParams.infileName, "rb");
  VERIFY(source, "Could not open input file");

  sink = fopen(OUTPUT_FILE, "wb");
  VERIFY(sink, "Could not create output file");

  // Initialize VPL session
  loader = MFXLoad();
  VERIFY(NULL != loader, "MFXLoad failed -- is implementation in path?");

  // Implementation used must be the type requested from command line
  cfg[0] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[0], "MFXCreateConfig failed")

  sts = MFXSetConfigFilterProperty(cfg[0], (mfxU8 *)"mfxImplDescription.Impl",
                                   cliParams.implValue);
  VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for Impl");

  // Implementation must provide VPP scaling
  cfg[1] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[1], "MFXCreateConfig failed")
  cfgVal[1].Type = MFX_VARIANT_TYPE_U32;
  cfgVal[1].Data.U32 = MFX_EXTBUFF_VPP_SCALING;
  sts = MFXSetConfigFilterProperty(
      cfg[1],
      (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
      cfgVal[1]);
  VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed");

  // Implementation used must provide API version 2.2 or newer
  cfg[2] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[2], "MFXCreateConfig failed")
  cfgVal[2].Type = MFX_VARIANT_TYPE_U32;
  cfgVal[2].Data.U32 =
      VPLVERSION(MAJOR_API_VERSION_REQUIRED, MINOR_API_VERSION_REQUIRED);
  sts = MFXSetConfigFilterProperty(
      cfg[2], (mfxU8 *)"mfxImplDescription.ApiVersion.Version", cfgVal[2]);
  VERIFY(MFX_ERR_NONE == sts,
         "MFXSetConfigFilterProperty failed for API version");

  sts = MFXCreateSession(loader, 0, &session);
  VERIFY(MFX_ERR_NONE == sts,
         "Cannot create session -- no implementations meet selection criteria");

  // Print info about implementation loaded
  ShowImplementationInfo(loader, 0);

  // Convenience function to initialize available accelerator(s)
  accelHandle = InitAcceleratorHandle(session);

  // Initialize VPP parameters
  if (MFX_IMPL_SOFTWARE == cliParams.impl) {
    PrepareFrameInfo(&VPPParams.vpp.In, MFX_FOURCC_I420, cliParams.srcWidth,
                     cliParams.srcHeight);
    PrepareFrameInfo(&VPPParams.vpp.Out, MFX_FOURCC_BGRA, OUTPUT_WIDTH,
                     OUTPUT_HEIGHT);
    VPPParams.IOPattern =
        MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
  } else {
    PrepareFrameInfo(&VPPParams.vpp.In, MFX_FOURCC_NV12, cliParams.srcWidth,
                     cliParams.srcHeight);
    PrepareFrameInfo(&VPPParams.vpp.Out, MFX_FOURCC_BGRA, OUTPUT_WIDTH,
                     OUTPUT_HEIGHT);
    VPPParams.IOPattern =
        MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
  }

  blur_data = (mfxU8 *)calloc(OUTPUT_WIDTH * OUTPUT_HEIGHT * 4, 1);

  // Initialize VPP
  sts = MFXVideoVPP_Init(session, &VPPParams);
  VERIFY(MFX_ERR_NONE == sts, "Could not initialize VPP");

  printf("Processing %s -> %s\n", cliParams.infileName, OUTPUT_FILE);

  while (isStillGoing == true) {
    // Load a new frame if not draining
    if (isDraining == false) {
      sts = MFXMemory_GetSurfaceForVPPIn(session, &vppInSurface);
      VERIFY(MFX_ERR_NONE == sts,
             "Unknown error in MFXMemory_GetSurfaceForVPPIn");

      sts = ReadRawFrame_InternalMem(vppInSurface, source);
      if (sts == MFX_ERR_MORE_DATA)
        isDraining = true;
      else
        VERIFY(MFX_ERR_NONE == sts, "Unknown error reading input");

      sts = MFXMemory_GetSurfaceForVPPOut(session, &vppOutSurface);
      VERIFY(MFX_ERR_NONE == sts,
             "Unknown error in MFXMemory_GetSurfaceForVPPIn");
    }

    sts = MFXVideoVPP_RunFrameVPPAsync(
        session, (isDraining == true) ? NULL : vppInSurface, vppOutSurface,
        NULL, &syncp);

    if (!isDraining) {
      sts_r = vppInSurface->FrameInterface->Release(vppInSurface);
      VERIFY(MFX_ERR_NONE == sts_r, "mfxFrameSurfaceInterface->Release failed");
    }

    switch (sts) {
      case MFX_ERR_NONE:
        do {
          sts = vppOutSurface->FrameInterface->Synchronize(
              vppOutSurface, WAIT_100_MILLISECONDS);
          if (MFX_ERR_NONE == sts) {
#ifdef BUILD_DPCPP

            if (MFX_IMPL_HARDWARE == cliParams.impl) {
              // map
              VADisplay va_display;
              mfxHandleType device_type;
              mfxHDL resource;
              mfxResourceType resource_type;
              mfxStatus sts;
              sts = vppOutSurface->FrameInterface->GetDeviceHandle(
                  vppOutSurface, &va_display, &device_type);
              // printf("GetDeviceHandle sts=%d\n",sts);
              sts = vppOutSurface->FrameInterface->GetNativeHandle(
                  vppOutSurface, &resource, &resource_type);
              // printf("GetNativeHandle sts=%d\n",sts);

              vaapiMemId *myMID = (vaapiMemId *)(vppOutSurface->Data.MemId);
              vaapiMemId *vm = (vaapiMemId *)resource;
              VASurfaceID va_surface_id;
              va_surface_id = *(VASurfaceID *)resource;
              // printf("myMID=%p resource=%p  %u\n",myMID,resource,
              // va_surface_id);

              vaSyncSurface(va_display, va_surface_id);
              usm_image_context context;
              VADRMPRIMESurfaceDescriptor prime_desc{};
              printf("vaExportSurfaceHandle surface ID=%u\n", va_surface_id);
              vaExportSurfaceHandle(va_display, va_surface_id,
                                    VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                                    VA_EXPORT_SURFACE_READ_ONLY, &prime_desc);
              int dma_fd = prime_desc.objects[0].fd;
              size_t dma_size = prime_desc.objects[0].size;
              context.drm_format_modifier =
                  prime_desc.objects[0]
                      .drm_format_modifier;  // non-zero if tiled (non-linear)
                                             // mem

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
#ifndef USE_LEVEL_ZERO_IPC_WORKAROUND
              // use this version if available -- it is the preferred approach
              ze_device_mem_alloc_desc_t alloc_desc = {};
              alloc_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
              ze_external_memory_import_fd_t import_fd = {
                  ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                  nullptr,  // pNext
                  ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, dma_fd};
              alloc_desc.pNext = &import_fd;
              ze_res = zeMemAllocDevice(ze_context, &alloc_desc, dma_size, 0,
                                        ze_device, &ptr);
#else
              // Use IPC workaround until approach above more widely supported
              // in level-zero
              ze_ipc_mem_handle_t ipc_handle{};
              *(size_t *)ipc_handle.data = dma_fd;
              ze_res = zeMemOpenIpcHandle(ze_context, ze_device, ipc_handle, 0,
                                          &ptr);
#endif
              if (ze_res != ZE_RESULT_SUCCESS) {
                throw std::runtime_error("Failed to get USM pointer");
              }
            }
            simple_pixel_flip(q, vppOutSurface->Data.B, blur_data,
                              OUTPUT_WIDTH * OUTPUT_HEIGHT * 4);

            // Write to output file
            {
              mfxFrameSurface1 blurred_surface{};
              blurred_surface.Info = VPPParams.vpp.Out;
              blurred_surface.Data.B = blur_data;
              blurred_surface.Data.G = blurred_surface.Data.B + 1;
              blurred_surface.Data.R = blurred_surface.Data.G + 1;
              blurred_surface.Data.A = blurred_surface.Data.R + 1;
              blurred_surface.Data.Pitch = VPPParams.vpp.Out.Width * 4;

              sts = WriteRawFrame(&blurred_surface, sink);

              if (sts != MFX_ERR_NONE) {
                printf("Error in WriteRawFrame\n");
                return sts;
              }
            }
            vppOutSurface->FrameInterface->Release(vppOutSurface);
#else
            sts = WriteRawFrame_InternalMem(vppOutSurface, sink);
            VERIFY(MFX_ERR_NONE == sts, "Could not write vpp output");
#endif
            framenum++;
          }
        } while (sts == MFX_WRN_IN_EXECUTION);
        break;
      case MFX_ERR_MORE_DATA:
        // Need more input frames before VPP can produce an output
        if (isDraining) isStillGoing = false;
        break;
      case MFX_ERR_MORE_SURFACE:
        // Need more surfaces at output for additional output frames available.
        // This applies to external memory allocations and should not be
        // expected for a simple internal allocation case like this
        break;
      case MFX_ERR_DEVICE_LOST:
        // For non-CPU implementations,
        // Cleanup if device is lost
        break;
      case MFX_WRN_DEVICE_BUSY:
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
  printf("Processed %d frames\n", framenum);

  // Clean up resources - It is recommended to close components first, before
  // releasing allocated surfaces, since some surfaces may still be locked by
  // internal resources.
  if (source) fclose(source);

  if (sink) fclose(sink);

  MFXVideoVPP_Close(session);
  MFXClose(session);

  if (accelHandle) FreeAcceleratorHandle(accelHandle);

  if (loader) MFXUnload(loader);

  if (MFX_IMPL_SOFTWARE == cliParams.impl) {
    free(blur_data);
  } else {
    sycl::free(blur_data, q);
  }

  return 0;
}
