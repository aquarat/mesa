/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_wsi.h"
#include "hk_entrypoints.h"
#include "hk_device.h"
#include "hk_instance.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "wsi_common.h"
#include "util/u_atomic.h"

#include <xf86drm.h>

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
hk_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *pName)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   return vk_instance_get_proc_addr_unchecked(pdev->vk.instance, pName);
}

static bool
hk_wsi_can_present_on_device(VkPhysicalDevice physicalDevice, int fd)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   if (pdev->dev.is_virtio) {
      return false;
   }

   drmDevicePtr device;
   if (drmGetDevice2(fd, 0, &device) != 0) {
      return false;
   }

   /* Allow on-device presentation for all non-virtio devices with bus type
    * PLATFORM */
   bool match = device->bustype == DRM_BUS_PLATFORM;

   drmFreeDevice(&device);

   return match;
}

VkResult
hk_init_wsi(struct hk_physical_device *pdev)
{
   VkResult result;

   struct wsi_device_options wsi_options = {.sw_device = false};
   result = wsi_device_init(
      &pdev->wsi_device, hk_physical_device_to_handle(pdev), hk_wsi_proc_addr,
      &pdev->vk.instance->alloc, pdev->master_fd,
      &hk_physical_device_instance(pdev)->drirc.options, &wsi_options);
   if (result != VK_SUCCESS)
      return result;

   pdev->wsi_device.supports_scanout = false;
   pdev->wsi_device.supports_modifiers = true;
   pdev->wsi_device.can_present_on_device = hk_wsi_can_present_on_device;

   pdev->vk.wsi_device = &pdev->wsi_device;

   return result;
}

void
hk_finish_wsi(struct hk_physical_device *pdev)
{
   pdev->vk.wsi_device = NULL;
   wsi_device_finish(&pdev->wsi_device, &pdev->vk.instance->alloc);
}

/*
 * Overriding the common entrypoint purely to count frames. Nothing else on
 * Asahi can tell us the frame rate once the FEX Vulkan thunk is active: an
 * x86-64 MangoHud is loaded by the guest loader, which the thunk replaces, and
 * the host-side aarch64 MangoHud is not visible inside the pressure-vessel
 * mount namespace. Counting presents in the driver sidesteps both problems and
 * puts the frame rate in the same report as the GPU timing it has to be read
 * against.
 */
VKAPI_ATTR VkResult VKAPI_CALL
hk_QueuePresentKHR(VkQueue _queue, const VkPresentInfoKHR *pPresentInfo)
{
   struct vk_queue *vk_queue = vk_queue_from_handle(_queue);
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   struct hk_device *dev = hk_queue_device(queue);

   VkResult result = wsi_common_queue_present(
      &hk_device_physical(dev)->wsi_device, &queue->vk, pPresentInfo);

   /* SUBOPTIMAL still put a frame on screen. */
   if (dev->gputime.enabled &&
       (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR))
      p_atomic_inc(&dev->gputime.presents);

   return result;
}
