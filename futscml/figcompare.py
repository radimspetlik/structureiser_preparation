import cv2
import numpy as np

def make_comparison_figure(images, labels, labelpos='top', labelcolor=(0.,0.,0.), bgcolor=(255, 255, 255), fontscale=0.75, h_label_offset=5,
                              font=cv2.FONT_HERSHEY_TRIPLEX, font_thickness=1, wspace_between_imgs=20):
    num_imgs = len(images)
    assert labelpos in ['top', 'inside'], "Other labelpos NYI"
    
    max_text_w = 0.
    max_text_h = 0.
    max_baseline = 0
    imgs_w_sum = 0
    imgs_h_max = 0
    imgs_c_max = 0
    for im in range(num_imgs):
        (label_width_actual, label_height_actual), baseline_actual = cv2.getTextSize(labels[im], font, fontscale, font_thickness)
        if label_width_actual > max_text_w:
            max_text_w = label_width_actual
        if label_height_actual + baseline_actual > max_text_h:
            max_text_h = label_height_actual + baseline_actual
        if images[im].shape[-3] > imgs_h_max:
            imgs_h_max = images[im].shape[-3]
        if images[im].shape[-1] > imgs_c_max:
            imgs_c_max = images[im].shape[-1]
        if baseline_actual > max_baseline:
            max_baseline = baseline_actual
        imgs_w_sum += images[im].shape[-2]
        
    canvas = np.ones((imgs_h_max + ((max_text_h + 5) if labelpos == 'top' else 0), imgs_w_sum + (num_imgs - 1) * wspace_between_imgs, imgs_c_max), dtype=np.uint8) \
                * np.array(bgcolor).astype(np.uint8)
    
    # render images
    positions = []
    for im in range(num_imgs):
        offset = (wspace_between_imgs + positions[-1][1]) if len(positions) > 0 else 0
        positions.append((offset, offset + images[im].shape[-2]))
        htop = canvas.shape[-3] - images[im].shape[-3]
        canvas[htop:, offset:offset+images[im].shape[-2], :] = images[im].astype(np.uint8) if images[im].shape[-1] == canvas.shape[-1] else \
                                                                np.concatenate((images[im].astype(np.uint8), np.ones_like(images[im], dtype=np.uint8)[:, :, canvas.shape[-1]-images[im].shape[-1]] * 255), axis=2)
    
    # render labels
    for im in range(num_imgs):
        hpos = canvas.shape[-3] - images[im].shape[-3] - max_baseline if labelpos=='top' else canvas.shape[-3] - images[im].shape[-3] + max_text_h
        position = (positions[im][0] + h_label_offset, hpos)
        canvas = cv2.putText(canvas, labels[im], position, font, fontscale, labelcolor, font_thickness, cv2.LINE_AA)
    
    return canvas
    