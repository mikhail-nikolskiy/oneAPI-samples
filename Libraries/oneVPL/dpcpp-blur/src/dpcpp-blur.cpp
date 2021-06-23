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

#include "util.h"
#include "vaapi_allocator.h"

#include <level_zero/ze_api.h>
#include <CL/sycl.hpp>
#include <CL/sycl/backend.hpp>
#include <CL/sycl/backend/level_zero.hpp>
#include <va/va.h>
#include <va/va_vpp.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
#include <unistd.h>
#include <drm/drm_fourcc.h>


#define OUTPUT_WIDTH 640
#define OUTPUT_HEIGHT 480
#define OUTPUT_FILE "out.raw"

#define MAX_PLANES_NUMBER 4


struct usm_image_context
{
  ze_context_handle_t ze_context;
  void *ptr;
  uint64_t drm_format_modifier;

  uint32_t planes_count;

  uint32_t offset[MAX_PLANES_NUMBER];
  uint32_t pitch[MAX_PLANES_NUMBER];
};



void simple_pixel_flip(cl::sycl::queue deviceQueue, mfxU8 *VA, mfxU8 *VC,
                       unsigned long N)
{
  constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
  constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

  cl::sycl::buffer<mfxU8, 1> bufferA(VA, cl::sycl::range<1>(N));
  cl::sycl::buffer<mfxU8, 1> bufferC(VC, cl::sycl::range<1>(N));

  deviceQueue.submit([&](cl::sycl::handler &cgh)
                     {
                       auto accessorA = bufferA.template get_access<sycl_read>(cgh);
                       auto accessorC = bufferC.template get_access<sycl_write>(cgh);

                       auto kern = [=](cl::sycl::id<1> wiID)
                       {
                         accessorC[wiID] = 255 - accessorA[wiID];
                       };
                       cgh.parallel_for(cl::sycl::range<1>(N), kern);
                     });
}



void Usage(void)
{
    printf("\n");
    printf("   Usage  :  legacy-vpp\n");
    printf("     -hw        use hardware implementation\n");
    printf("     -sw        use software implementation\n");
    printf("     -i input file name (sw=I420 raw frames, hw=NV12)\n");
    printf("     -w input width\n");
    printf("     -h input height\n\n");
    printf("   Example:  legacy-vpp -i in.i420 -w 128 -h 96\n");
    printf(
        "   To view:  ffplay -f rawvideo -pixel_format yuv420p -video_size %dx%d -pixel_format yuv420p %s\n\n",
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
        OUTPUT_FILE);
    printf(" * Resize raw frames to %dx%d size in %s\n\n",
           OUTPUT_WIDTH,
           OUTPUT_HEIGHT,
           OUTPUT_FILE);
    return;
}

int main(int argc, char **argv)
{
    sycl::queue q = sycl::gpu_selector{};
    mfxU8 *out_data = (mfxU8 *)calloc(1920 * 1088 * 2, 1);  //update with current resolution

    mfxStatus sts = MFX_ERR_NONE;
    bool isDraining = false;
    bool isStillGoing = true;
    FILE *sink = NULL;
    FILE *source = NULL;
    //mfxFrameSurface1 *vppInSurface  = NULL;
    mfxFrameSurface1 *vppOutSurface = NULL;
    mfxU32 framenum = 0;

    mfxFrameSurface1 *vppInSurfacePool = NULL;
    mfxFrameSurface1 *vppOutSurfacePool = NULL;
    Params cliParams = {};

    //Parse command line args to cliParams
    if (ParseArgsAndValidate(argc, argv, &cliParams, PARAMS_VPP) == false)
    {
        Usage();
        return 1; // return 1 as error code
    }

    source = fopen(cliParams.infileName, "rb");

    sink = fopen(OUTPUT_FILE, "wb");

    mfxVersion ver = {{0, 1}};
    mfxSession session;

    mfxFrameAllocator mfxAllocator;
    mfxFrameAllocator *pmfxAllocator = &mfxAllocator;

    // Initialize Intel Media SDK Session

    sts = MFXInit(cliParams.impl, &ver, &session);

    // open VA display, set handle, and set allocator

    VAStatus va_res = VA_STATUS_SUCCESS;
    int major_version = 0, minor_version = 0;

    m_va_dpy = (VADisplay)InitAcceleratorHandle(session);

    // Provide VA display handle to Media SDK
    sts = MFXVideoCORE_SetHandle(session, static_cast<mfxHandleType>(MFX_HANDLE_VA_DISPLAY),
                                 m_va_dpy);

    // If mfxFrameAllocator is provided it means we need to setup  memory allocator
    if (pmfxAllocator)
    {
        pmfxAllocator->pthis = &session; // We use Media SDK session ID as the allocation identifier
        pmfxAllocator->Alloc = simple_alloc;
        pmfxAllocator->Free = simple_free;
        pmfxAllocator->Lock = simple_lock;
        pmfxAllocator->Unlock = simple_unlock;
        pmfxAllocator->GetHDL = simple_gethdl;

        // Since we are using video memory we must provide Media SDK with an external allocator
        sts = MFXVideoCORE_SetFrameAllocator(session, pmfxAllocator);
    }

    // Initialize VPP parameters
    // - For video memory surfaces are used to store the raw frames
    //   (Note that when using HW acceleration video surfaces are prefered, for better performance)
    mfxVideoParam VPPParams;
    memset(&VPPParams, 0, sizeof(VPPParams));
    // Input data
    VPPParams.vpp.In.FourCC = MFX_FOURCC_NV12;
    VPPParams.vpp.In.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    VPPParams.vpp.In.CropX = 0;
    VPPParams.vpp.In.CropY = 0;
    VPPParams.vpp.In.CropW = cliParams.srcWidth;
    VPPParams.vpp.In.CropH = cliParams.srcHeight;
    VPPParams.vpp.In.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    VPPParams.vpp.In.FrameRateExtN = 30;
    VPPParams.vpp.In.FrameRateExtD = 1;
    // width must be a multiple of 16
    // height must be a multiple of 16 in case of frame picture and a multiple of 32 in case of field picture
    VPPParams.vpp.In.Width = ALIGN16(VPPParams.vpp.In.CropW);
    VPPParams.vpp.In.Height = ALIGN16(VPPParams.vpp.In.CropH);

    // Output data
    VPPParams.vpp.Out.FourCC = MFX_FOURCC_NV12;
    VPPParams.vpp.Out.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    VPPParams.vpp.Out.CropX = 0;
    VPPParams.vpp.Out.CropY = 0;
    VPPParams.vpp.Out.CropW = OUTPUT_WIDTH;
    VPPParams.vpp.Out.CropH = OUTPUT_HEIGHT;
    VPPParams.vpp.Out.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    VPPParams.vpp.Out.FrameRateExtN = 30;
    VPPParams.vpp.Out.FrameRateExtD = 1;
    // width must be a multiple of 16
    // height must be a multiple of 16 in case of frame picture and a multiple of 32 in case of field picture
    VPPParams.vpp.Out.Width = ALIGN16(VPPParams.vpp.Out.CropW);
    VPPParams.vpp.Out.Height = ALIGN16(VPPParams.vpp.Out.CropH);

    VPPParams.IOPattern =
        MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    // Initialize Media SDK VPP
    sts = MFXVideoVPP_Init(session, &VPPParams);

    // Query number of required surfaces for VPP
    mfxFrameAllocRequest VPPRequest[2]; // [0] - in, [1] - out
    memset(&VPPRequest, 0, sizeof(mfxFrameAllocRequest) * 2);
    sts = MFXVideoVPP_QueryIOSurf(session, &VPPParams, VPPRequest);

    // Allocate required surfaces
    mfxFrameAllocResponse mfxResponseIn;
    mfxFrameAllocResponse mfxResponseOut;
    sts = mfxAllocator.Alloc(mfxAllocator.pthis, &VPPRequest[0], &mfxResponseIn);

    sts = mfxAllocator.Alloc(mfxAllocator.pthis, &VPPRequest[1], &mfxResponseOut);

    mfxU16 nSurfNumVPPIn = mfxResponseIn.NumFrameActual;
    mfxU16 nSurfNumVPPOut = mfxResponseOut.NumFrameActual;

    // Allocate surface headers (mfxFrameSurface1) for VPP
    vppInSurfacePool = (mfxFrameSurface1 *)calloc(sizeof(mfxFrameSurface1), nSurfNumVPPIn);
    for (int i = 0; i < nSurfNumVPPIn; i++)
    {
        memset(&vppInSurfacePool[i], 0, sizeof(mfxFrameSurface1));
        vppInSurfacePool[i].Info = VPPParams.vpp.In;
        vppInSurfacePool[i].Data.MemId = mfxResponseIn.mids[i]; // MID (memory id) represent one D3D NV12 surface
    }

    vppOutSurfacePool = (mfxFrameSurface1 *)calloc(sizeof(mfxFrameSurface1), nSurfNumVPPOut);
    for (int i = 0; i < nSurfNumVPPOut; i++)
    {
        memset(&vppOutSurfacePool[i], 0, sizeof(mfxFrameSurface1));
        vppOutSurfacePool[i].Info = VPPParams.vpp.Out;
        vppOutSurfacePool[i].Data.MemId = mfxResponseOut.mids[i]; // MID (memory id) represent one D3D NV12 surface
    }

    // ===================================
    // Start processing the frames
    //

    int nIndexVPPInSurf = 0, nIndexVPPOutSurf = 0;
    mfxSyncPoint syncp;
    mfxU32 nFrame = 0;

    printf("Processing %s -> %s\n", cliParams.infileName, OUTPUT_FILE);

    while (isStillGoing == true)
    {

        if (isDraining == false)
        {
            nIndexVPPInSurf = GetFreeSurfaceIndex(vppInSurfacePool, nSurfNumVPPIn); // Find free input frame surface

            sts = mfxAllocator.Lock(mfxAllocator.pthis, vppInSurfacePool[nIndexVPPInSurf].Data.MemId, &(vppInSurfacePool[nIndexVPPInSurf].Data));
            //verify

            sts = ReadRawFrame(&vppInSurfacePool[nIndexVPPInSurf], source); // Load frame from file into surface
            printf("ReadRawFrame sts=%d\n", sts);
            if (sts == MFX_ERR_MORE_DATA)
                isDraining = true;
            //else
            //VERIFY(MFX_ERR_NONE == sts, "Unknown error reading input");

            sts = mfxAllocator.Unlock(mfxAllocator.pthis, vppInSurfacePool[nIndexVPPInSurf].Data.MemId, &(vppInSurfacePool[nIndexVPPInSurf].Data));
            //verify
        }

        nIndexVPPOutSurf = GetFreeSurfaceIndex(vppOutSurfacePool, nSurfNumVPPOut); // Find free output frame surface

        sts = MFXVideoVPP_RunFrameVPPAsync(session,
                                           (isDraining == true) ? NULL : &vppInSurfacePool[nIndexVPPInSurf],
                                           &vppOutSurfacePool[nIndexVPPOutSurf], NULL, &syncp);
        printf("VPP sts=%d\n", sts);

        switch (sts)
        {
        case MFX_ERR_NONE:
        {
            sts = MFXVideoCORE_SyncOperation(session, syncp, WAIT_100_MILLISECONDS * 1000);
            //verify

            // Surface locking required when read/write video surfaces
            sts = mfxAllocator.Lock(mfxAllocator.pthis, vppOutSurfacePool[nIndexVPPOutSurf].Data.MemId, &(vppOutSurfacePool[nIndexVPPOutSurf].Data));
            //verify

            //sts = WriteRawFrame(&vppOutSurfacePool[nIndexVPPOutSurf], sink);
            //verify

#if 1        
        mfxFrameSurface1 *pmfxOutSurface;
    pmfxOutSurface = &vppOutSurfacePool[nIndexVPPOutSurf];

    vaapiMemId *myMID = (vaapiMemId *)(pmfxOutSurface->Data.MemId);
    printf("Res= %dx%d,  VASurfaceID*=%p\n", pmfxOutSurface->Info.CropW,
           pmfxOutSurface->Info.CropH, myMID->m_surface);

    //V.lockSurf(pmfxOutSurface);

    VASurfaceID va_surface_id = *myMID->m_surface;
    usm_image_context context;
    VADRMPRIMESurfaceDescriptor prime_desc={};
    printf("vaExportSurfaceHandle surface ID=%u\n", va_surface_id);
    vaExportSurfaceHandle(m_va_dpy, va_surface_id, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                          VA_EXPORT_SURFACE_READ_ONLY, &prime_desc);
    int dma_fd = prime_desc.objects[0].fd;
    size_t dma_size = prime_desc.objects[0].size;
    context.drm_format_modifier = prime_desc.objects[0].drm_format_modifier; // non-zero if tiled (non-linear) mem

    uint32_t n_planes = 0;
    for (uint32_t i = 0; i < prime_desc.num_layers; i++)
    {
      auto layer = &prime_desc.layers[i];
      for (uint32_t j = 0; j < layer->num_planes; j++)
      {
        if (n_planes < MAX_PLANES_NUMBER)
        {
          context.pitch[n_planes] = layer->pitch[j];
          context.offset[n_planes] = layer->offset[j];
          n_planes++;
        }
      }
    }
    context.planes_count = n_planes;

    ze_context_handle_t ze_context = q.get_context().get_native<sycl::backend::level_zero>();
    context.ze_context = ze_context;
    ze_device_handle_t ze_device = q.get_device().get_native<sycl::backend::level_zero>();
    void *ptr = nullptr;
    ze_result_t ze_res;
#if 0
    //use this version -- it is the preferred approach
    ze_device_mem_alloc_desc_t alloc_desc = {};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    ze_external_memory_import_fd_t import_fd = {
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
            nullptr, // pNext
            ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
            dma_fd
    };
    alloc_desc.pNext = &import_fd;
    ze_res = zeMemAllocDevice(ze_context, &alloc_desc, dma_size, 0, ze_device, &ptr);
#else
    // TODO: use zeMemAllocDevice call above, once it supported in level-zero
    ze_ipc_mem_handle_t ipc_handle{};
    *(size_t *)ipc_handle.data = dma_fd;
    ze_res = zeMemOpenIpcHandle(ze_context, ze_device, ipc_handle, 0, &ptr);
    printf("ze_context=%p ze_device=%p %d\n",ze_context,ze_device,dma_fd);
#endif
    if (ze_res != ZE_RESULT_SUCCESS)
    {
        printf("ze_res=%0X\n",ze_res);
      throw std::runtime_error("Failed to get USM pointer");
    }

    printf("%p prime_desc fourcc=%04X (%dx%d) num_layers=%d\n", ptr, prime_desc.fourcc, prime_desc.width, prime_desc.height, prime_desc.num_layers);
    for (uint32_t i = 0; i < prime_desc.num_layers; i++)
    {
      auto layer = &prime_desc.layers[i];
      printf(" layer %d has %d planes\n", i, layer->num_planes);
      for (uint32_t j = 0; j < layer->num_planes; j++)
      {
        printf(" plane %d layer %d offset=%d pitch=%d\n", i, j, layer->offset[j], layer->pitch[j]);
      }
    }

    close(dma_fd);
    context.ptr = ptr;

    mfxU8 *y_in_ptr = pmfxOutSurface->Data.Y;
    mfxU8 *uv_in_ptr = pmfxOutSurface->Data.UV;
    mfxU32 pitch = pmfxOutSurface->Data.Pitch;

    int W = prime_desc.width;
    int H = prime_desc.height;
    mfxU32 vector_length = W * H * 1.5;

#if 1
    printf("pixel flip inline\n");
    for (int i=0;i<W*H*1.5;i++) {
      out_data[i]=(mfxU8)(255-y_in_ptr[i]);
    }
#else

    simple_pixel_flip(q, y_in_ptr, out_data, W * H * 1.5);

#endif

    for (int r = 0; r < H; r++)
    {
      fwrite(out_data + (r * pitch), 1, W,
             sink);
    }

    for (int r = 0; r < H * .5; r++)
    {
      fwrite(out_data + prime_desc.layers[1].offset[0] + (r * pitch), 1, W, sink);
    }
#endif


            sts = mfxAllocator.Unlock(mfxAllocator.pthis, vppOutSurfacePool[nIndexVPPOutSurf].Data.MemId, &(vppOutSurfacePool[nIndexVPPOutSurf].Data));
            //verify

            printf("Frame number: %d\n", ++framenum);
            fflush(stdout);
        }
            break;
        case MFX_ERR_MORE_DATA:
            // Need more input frames before VPP can produce an output
            if (isDraining)
                isStillGoing = false;
            break;
        case MFX_ERR_MORE_SURFACE:
            // The output frame is ready after synchronization.
            // Need more surfaces at output for additional output frames available.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
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

    mfxAllocator.Free(mfxAllocator.pthis, &mfxResponseIn);
    mfxAllocator.Free(mfxAllocator.pthis, &mfxResponseOut);

    //Release();

    return 0;
}
