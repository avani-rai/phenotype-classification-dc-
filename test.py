#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import shutil

def attribute_merging(pixels, coordinates):

    atts = {
        0: 'bg',
        1: 'skin',
        2: 'l_brow',
        3: 'r_brow',
        4: 'l_eye',
        5: 'r_eye',
        6: 'eye_g',
        7: 'l_ear',
        8: 'r_ear',
        9: 'ear_r',
        10: 'nose',
        11: 'mouth',
        12: 'u_lip',
        13: 'l_lip',
        14: 'neck',
        15: 'neck_l',
        16: 'cloth',
        17: 'hair',
        18: 'hat'
    }

    skin_indexes = [1, 10, 14]
    brow_indexes = [2, 3]
    eye_indexes = [4, 5, 6]
    ear_indexes = [7, 8, 9]
    lip_indexes = [12, 13]
    hair_indexes = [17]

    skin_arr = []  # stores color (skin + nose + neck)
    brow_arr = []  # stores coordinates (l_brow + r_brow)
    eye_arr = []   # stores color (l_eye + r_eye + eye_g)
    ear_arr = []   # stores coordinates (l_ear + r_ear + ear_r)
    lip_arr = []   # stores color (u_lip + l_lip)
    hair_arr = []  # stores color (hair)

    for index in range(len(atts)):
      if index in skin_indexes:
        skin_arr.extend(pixels[index])
      elif index in brow_indexes:
        brow_arr.extend(coordinates[index])
      elif index in eye_indexes:
        eye_arr.extend(pixels[index])
      elif index in ear_indexes:
        ear_arr.extend(coordinates[index])
      elif index in lip_indexes:
        lip_arr.extend(pixels[index])
      elif index in hair_indexes:
        hair_arr.extend(pixels[index])

    return skin_arr, brow_arr, eye_arr, ear_arr, lip_arr, hair_arr


def vis_parsing_maps(
    im,
    parsing_anno,
    stride,
    save_im=False,
    save_path="vis_results/parsing_map_on_im.jpg",
):
    # Colors for all 20 parts
    part_colors = [
        [0, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [255, 255, 170],
        [0, 0, 0],
        [255, 85, 255],
        [255, 170, 255],
        [0, 255, 255],
        [85, 255, 255],
        [170, 255, 255],
    ]

    atts = {
        0: 'bg',
        1: 'skin',
        2: 'l_brow',
        3: 'r_brow',
        4: 'l_eye',
        5: 'r_eye',
        6: 'eye_g',
        7: 'l_ear',
        8: 'r_ear',
        9: 'ear_r',
        10: 'nose',
        11: 'mouth',
        12: 'u_lip',
        13: 'l_lip',
        14: 'neck',
        15: 'neck_l',
        16: 'cloth',
        17: 'hair',
        18: 'hat'
    }


    pixel_array = []
    pixel_coordinate = []

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST
    )
    vis_parsing_anno_color = (
        np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    )

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, 19):
        index = np.where(vis_parsing_anno == pi)
        pixel_coordinate.append(np.transpose(index))
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        temp_img = (
            np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
        )

        temp_img[index[0], index[1], :] = part_colors[pi]
        temp_img = temp_img.astype(np.uint8)
        temp_img = cv2.addWeighted(
            cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, temp_img, 0.6, 0
        )
        attr = atts[pi]
        
        if save_im:
            cv2.imwrite(save_path[:-5] + f"_{attr}.png", temp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        temp_img = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
        pixel_values = temp_img[index[0], index[1], :]
        pixel_array.append(pixel_values)
        print(pi, " : ")
        print(pixel_values)
        # print(np.shape(pixel_values))
        print()

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(
        cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0
    )

    # Save result or not
    
    if save_im:
        cv2.imwrite(save_path[:-5] + ".png", vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return pixel_array, pixel_coordinate
    # return vis_im


def evaluate(respth="./res/test_res", dspth="./data", cp="pretrained_model.pth"):

    if os.path.exists(respth):
        shutil.rmtree(respth)
    os.makedirs(respth)

    base_dir = "/content/face_masking/"
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(base_dir, "models", cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            print(image_path)
            print(parsing)
            print(np.unique(parsing))
            print()

            # Create a subfolder with the same name as the input image
            input_image_name = osp.splitext(image_path)[0]  # Remove the file extension
            subfolder_path = osp.join(respth, input_image_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            img_pixel_array, img_pixel_coordinate = vis_parsing_maps(
                image,
                parsing,
                stride=1,
                save_im=True,
                save_path=osp.join(subfolder_path, image_path),
            )

            skin_arr, brow_arr, eye_arr, ear_arr, lip_arr, hair_arr = attribute_merging(img_pixel_array, img_pixel_coordinate)

            im2 = np.array(image)
            vis_im2 = im2.copy().astype(np.uint8)
            vis_im2 = cv2.cvtColor(vis_im2, cv2.COLOR_RGB2BGR)

            if brow_arr:
                mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                for coordinate in brow_arr:
                    mask[coordinate[0], coordinate[1]] = vis_im2[coordinate[0], coordinate[1]]
                cv2.imwrite(osp.join(subfolder_path, f"brow_mask.png"), mask)

            if ear_arr:
                mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

                for coordinate in ear_arr:
                    mask[coordinate[0], coordinate[1]] = vis_im2[coordinate[0], coordinate[1]]

                cv2.imwrite(osp.join(subfolder_path, f"ear_mask.png"), mask)

            np.savetxt(osp.join(subfolder_path, 'skin_colors.txt'), skin_arr, fmt='%d')
            np.savetxt(osp.join(subfolder_path, 'eye_colors.txt'), eye_arr, fmt='%d')
            np.savetxt(osp.join(subfolder_path, 'lip_colors.txt'), lip_arr, fmt='%d')
            np.savetxt(osp.join(subfolder_path, 'hair_colors.txt'), hair_arr, fmt='%d')



            # print("length: ", len(img_pixel_array))

            # for i in img_pixel_coordinate:
            #   print(i)
            #   print()

            


if __name__ == "__main__":
    evaluate(
        dspth=f"/content/face_masking/data",
        cp="pretrained_model.pth",
    )