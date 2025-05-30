from patchify import patchify
import torch
import numpy as np
import cv2

def binarize(x, thresh, return_boolean=False):
    """NOT in-place."""
    if isinstance(x, np.ndarray):
        out = np.zeros_like(x)
    else:
        out = torch.zeros_like(x)
    mask_pos = (x > thresh)
    out[mask_pos] = 1.
    out[~mask_pos] = 0.
    if return_boolean:
        if isinstance(x, np.ndarray):
            return out.astype(bool)
        else:
            return out.bool()
    else:
        return out
    
def patchify_mask(mask, patch_size, target_size, device):
    """
    mask: ht, wd. np array
    
    return: 1, ht0, wd0. torch tensor
    """
    if patch_size is not None and target_size is not None:
        # ht0, wd0
        ht, wd = target_size
        mask = patchify(mask, (patch_size, patch_size), step=patch_size)
        mask = mask.reshape(ht, wd, -1).mean(-1)
    # c, ht, wd
    return torch.tensor(mask, dtype=torch.float32, device=device)[None]

def vis(img1: torch.Tensor, img2: torch.Tensor, alphamask1: np.ndarray, alphamask2: np.ndarray, dest=None):
    """Visualization, for splatting results."""
    img1, img2 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy(), img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    alphamask1 = (alphamask1 * 255).astype(np.uint8)
    alphamask2 = (alphamask2 * 255).astype(np.uint8)
    alphamask1 = cv2.applyColorMap(alphamask1, cv2.COLORMAP_BONE)
    alphamask2 = cv2.applyColorMap(alphamask2, cv2.COLORMAP_BONE)
    imgs = np.concatenate( [img1, img2], 0 )[:, :, [2,1,0]]
    alphamasks = np.concatenate( [alphamask1, alphamask2], 0 )
    blended = (imgs.astype(np.float32) * alphamasks) / 255
    final_image = np.concatenate( [imgs, blended, alphamasks], 1 )
    
    if dest is not None:
        cv2.imwrite(dest, final_image)

    return final_image

def save_visualization(img1, img2, out1, out2, dest):
    img1, img2 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy(), img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out1, out2 = out1.squeeze(0).squeeze(0).cpu().numpy(), out2.squeeze(0).squeeze(0).cpu().numpy()

    assert out1.shape == out2.shape and out1.shape in [(3, 4), (6, 8), (12, 16)]
    if out1.shape == (3, 4):
        pad_size = 64
    if out1.shape == (6, 8):
        pad_size = 32
    if out1.shape == (12, 16):
        pad_size = 16
    out1 = out1.repeat(pad_size, axis=0).repeat(pad_size, axis=1)
    out2 = out2.repeat(pad_size, axis=0).repeat(pad_size, axis=1)

    imgs = np.concatenate([img1, img2], axis=0)[:, :, [2, 1, 0]]
    outs = (np.concatenate([out1, out2], axis=0) * 255).astype(np.uint8)

    colormap = cv2.COLORMAP_BONE
    outs = cv2.applyColorMap(outs, colormap)

    imgs_and_outs = (imgs * outs) / 255.

    final_image = np.concatenate([imgs, imgs_and_outs, outs], axis=1)
    return cv2.imwrite(dest, final_image)