import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
import cv2

import annotations as annot

def get_random_niifilename(n: int = 1, folderpath: str) -> list:
    '''
    Get random filename of nii files from available in the challange.
    May be useful for plotting examples.
    Parameters:
    -----------
    n (int):
        number of nii files to randomly choose (default: 1)
    folderpath (str):
        path to folder where nii files are stored
    Returns:
    -----------
        list of length n with path to randomly chosen nii files
    '''
    flist = os.listdir(folderpath)
    random_files = [os.path.join(folderpath, random_file) for random_file in random.choices(flist, k=n)]
    return random_files


def load_niifile_as_image(filepath: str = None):
    if filepath is None:
        filepath = get_random_niifilename()[0]
    img = nib.load(filepath)
    return img


def load_niifile_as_numpy(filepath: str = None):
    if filepath is None:
        filepath = get_random_niifilename()[0]
    imgarray = nib.load(filepath).get_fdata()
    return imgarray


def show_multiple_slices(imgdata, n: int, axis: str = 'z', start_slice: int = 0, step = None):
    axs = ['x', 'y', 'z']
    axis_shape = imgdata.shape[axs.index(axis)]
    if step is None:
        step = (axis_shape - start_slice) // n

    # figure settings
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, n_rows*2))

    for i in range(n):
        if axis == 'x':
            ax.flat[i].imshow(imgdata[start_slice + i*step, :, :], cmap='bone')
        elif axis == 'y':
            ax.flat[i].imshow(imgdata[:, start_slice + i*step, :], cmap='bone')
        elif axis == 'z':
            ax.flat[i].imshow(imgdata[:, :, start_slice + i*step], cmap='bone')
        ax.flat[i].axis('off')
        ax.flat[i].set_title(f"{axis}={start_slice + i*step}")
    plt.tight_layout()
    plt.show()


def show_x_axis_with_z_line(filepath: str = None, n_fig: int = 5, 
                            step: int = 10, train_csv: str = 'train.csv'):
    
    # random file if filepath is None
    if filepath is None:
        filepath = get_random_niifilename()[0]
    # load data from .nii file
    imgdata = load_niifile_as_numpy(filepath)
    axis_shape = imgdata.shape[0]
    # load train.csv
    train_data = pd.read_csv(train_csv)
    
    # find 'Growth Plate Index' for given Image Name
    imgname = os.path.splitext(os.path.basename(filepath))[0]
    plate_index = train_data.loc[train_data['Image Name'] == imgname, 'Growth Plate Index'].values[0]
    
    # show n_fig' pictures with different x-coordinate that changes by 'step'
    axx = np.linspace(axis_shape // 2 - step * n_fig // 2, axis_shape // 2 + step * n_fig // 2, num=n_fig)
    
    # create figure
    fig, ax = plt.subplots(1, n_fig, figsize=(n_fig*3, 3))
    fig.suptitle(f"File: {imgname}.nii")
    for i, x in enumerate(axx):
        ax.flat[i].imshow(imgdata[int(x), :, :], cmap='bone')
        ax.flat[i].axvline(plate_index, color="red", ls="--")
        ax.flat[i].axis('off')
        ax.flat[i].set_title(f"x={int(x)}")
    plt.tight_layout()
    plt.show()
    
    
    
def show_target_plate(filepath: str = None, step: int = 5, n_fig: int = 5, train_csv: str = 'train.csv'):

    if not n_fig % 2:
        raise ValueError("Number of figures (n_fig) should be an odd number.")

    if filepath is None:
        filepath = get_random_niifilename()[0]
    # load data from  .nii file
    imgdata = load_niifile_as_numpy(filepath)
    # load train.csv
    train_data = pd.read_csv(train_csv)

    # find 'Growth Plate Index' for given Image Name
    imgname = os.path.splitext(os.path.basename(filepath))[0]
    axial_index = train_data.loc[train_data['Image Name'] == imgname, 'Growth Plate Index'].values[0]

    # show n_fig' pictures with set z-axis; with middle one z=axial_index (so target)
    axz = np.linspace(axial_index - step * n_fig // 2, axial_index + step * n_fig // 2, num=n_fig)
    
    fig, ax = plt.subplots(1, n_fig, figsize=(n_fig*3, 3))
    fig.suptitle(f"File: {imgname}.nii")
    for i, z in enumerate(axz):
        ax.flat[i].imshow(imgdata[:, :, int(z)], cmap='bone')
        ax.flat[i].axis('off')
        if z != axial_index:
            ax.flat[i].set_title(f"z={int(z)}")
        else:
            ax.flat[i].set_title(f"GROWTH PLATE INDEX = {int(z)}")
    plt.tight_layout()
    plt.show()

        
def show_target_plate_with_annot(filepath: str = None, step: int = 5, n_fig: int = 5, train_csv: str = 'train.csv'):

    if not n_fig % 2:
        raise ValueError("Number of figures (n_fig) should be an odd number.")

    if filepath is None:
        filepath = get_random_niifilename()[0]
    # load data from  .nii file
    imgdata = load_niifile_as_numpy(filepath)
    annot_mask = annot.create_annotation_mask(imgdata)
    # load train.csv
    train_data = pd.read_csv(train_csv)

    # find 'Growth Plate Index' for given Image Name
    imgname = os.path.splitext(os.path.basename(filepath))[0]
    axial_index = train_data.loc[train_data['Image Name'] == imgname, 'Growth Plate Index'].values[0]

    # show n_fig' pictures with set z-axis; with middle one z=axial_index (so target)
    axz = np.linspace(axial_index - step * n_fig // 2, axial_index + step * n_fig // 2, num=n_fig)
    
    fig, ax = plt.subplots(2, n_fig, figsize=(n_fig*3, 8))
    fig.suptitle(f"File: {imgname}.nii")
    for i, z in enumerate(axz):
        ax[0, i].imshow(imgdata[:, :, int(z)], cmap='bone')
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        if z != axial_index:
            ax[0,i].set_title(f"z={int(z)}")
        else:
            ax[0,i].set_title(f"GROWTH PLATE INDEX = {int(z)}")
        # with annotations
        im = ax[1, i].imshow(annot_mask[:, :, int(z)], cmap=annot.ANNOT_CMAP)
        
    # Create a common color bar for all subplots
    cbar_ax = fig.add_axes([0.1, 0.005, 0.8, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    num_ticks = 6
    cbar.set_ticks(np.linspace(5/5/2, 5 - 5/5/2, num_ticks))
    cbar.set_ticklabels(['UNKNOWN', 'AIR', 'FAT', 'BONE_CANCELLOUS', 'BONE_CORTICAL', 'WATER'])
    plt.tight_layout()
    plt.show()

    
def get_slices_sums_normalized(folderpath: str):
    flist = [os.path.join(folderpath, file) for file in os.listdir(folderpath)]
    
    sums = [[], [], []]
    arrays = []
    for file in tqdm(flist):
        imgarray = load_niifile_as_numpy(file)
        img_mask = np.logical_and(imgarray >= 0, imgarray <= 1900)
        for i in range(3):
            shapes = [img_mask.shape[j] for j in range(3) if j != 1]
            area = np.prod(shapes)
            sums_file = [np.sum(img_mask, axis=tuple(j for j in [0, 1, 2] if j != i)) / area for i in range(3)]
        for i, s in enumerate(sums_file):
            sums[i].append(s)
    for i, sum_list in tqdm(enumerate(sums)):
        biggest = max(sum_list, key=lambda x: x.shape[0])
        smallest = min(sum_list, key=lambda x: x.shape[0])
        arr = np.empty((len(sum_list), biggest.shape[0]))
        arr[:] = np.nan
        for j, element in enumerate(sum_list):
            if np.array_equal(biggest, element):
                print(f"Bigest array for file: {flist[j]} [{biggest.shape}] with index: {j}")
            if np.array_equal(smallest, element):
                print(f"Smallest array for file: {flist[j]} [{smallest.shape}] with index: {j}")
            arr[j, :element.shape[0]] = element
        arrays.append(arr)
        
    return arrays

    
def window_image(x_thr_min: float,
                 y_thr_min: float,
                 folderpath: str,
                 outputpath: str):
    '''
    Reduce amount of background.
    '''
    
    # make a folder
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    area_sums_norm = get_slices_sums_normalized(folderpath)
    thrs = [x_thr_min, y_thr_min]
    start_idxs = []
    end_idxs = []
    for i, ax_data in enumerate(area_sums_norm):
        max_per_slice = np.max(ax_data, axis=0)
        
        # for now skip z-axis
        if i == 2:
            continue
        
        start_idxs.append(np.where(max_per_slice > thrs[i])[0][0])
        end_idxs.append(np.where(max_per_slice > thrs[i])[0][-1])
        
    filenames = [f.split('.')[0] for f in os.listdir(folderpath)]
    flist = [os.path.join(folderpath, file) for file in os.listdir(folderpath)]
    for i, file in enumerate(flist):
        imgarray = load_niifile_as_numpy(file)
        print(imgarray.shape, 'original')
        imgarray_windowed = imgarray[start_idxs[0]:end_idxs[0], start_idxs[1]:end_idxs[1], :]
        print(imgarray_windowed.shape)
        np.save(os.path.join(outputpath, filenames[i] + '.npy'), imgarray_windowed)
        print(f"{filenames[i]}.npy saved")

def window_and_slice_image_combined(config):
    '''
    Automatically window the images and save them as individual slices.
    '''

    # Create the output folder if it doesn't exist
    if not os.path.exists(config['slices_data_dir']):
        os.makedirs(config['slices_data_dir'])

    # Calculate the normalized slice sums
    area_sums_norm = get_slices_sums_normalized(config['3d_images_dir'])
    thrs = [config['x_threshold_for_cropping'], config['y_threshold_for_cropping']]
    start_idxs = []
    end_idxs = []

    for i, ax_data in enumerate(area_sums_norm):
        max_per_slice = np.max(ax_data, axis=0)
        
        # Skip z-axis (index 2)
        if i == 2:
            continue

        # Determine the start and end indices for each dimension
        start_idxs.append(np.where(max_per_slice > thrs[i])[0][0])
        end_idxs.append(np.where(max_per_slice > thrs[i])[0][-1])

    # Process all files in the input folder
    filenames = [f.split('.')[0] for f in os.listdir(config['3d_images_dir']) if f.endswith('.nii') or f.endswith('.nii.gz')]
    flist = [os.path.join(config['3d_images_dir'], file) for file in os.listdir(config['3d_images_dir']) if file.endswith('.nii') or file.endswith('.nii.gz')]
    
    for i, file in enumerate(flist):
        imgarray = load_niifile_as_numpy(file)
        print(imgarray.shape, 'original')
        
        # Window the array
        xmin, xmax = start_idxs[0], end_idxs[0]
        ymin, ymax = start_idxs[1], end_idxs[1]
        imgarray_windowed = imgarray[xmin:xmax, ymin:ymax, :]
        # Ensure the windowing dimensions are valid
        if xmin < 0 or ymin < 0 or xmax > imgarray.shape[0] or ymax > imgarray.shape[1]:
            print(f"ERROR: Invalid window dimensions for {file}. Skipping.")
            continue
        
        # Save each slice along the z-axis
        for z in range(imgarray_windowed.shape[2]):
            slice_array = imgarray_windowed[:, :, z]
            slice_array = slice_array.reshape((slice_array.shape[0], slice_array.shape[1], 1))
            
            slice_filename = f"{filenames[i]}_{z}.npy"
            np.save(os.path.join(config['slices_data_dir'], slice_filename), slice_array)
            print(f"{slice_filename} saved")

        
def show_slices(file_code: str = None, axs: str = 'x', n_fig: int = 5, 
                step: int = 5, folderpath: str, train_csv: str):
    
    if not n_fig % 2:
        raise ValueError("Number of figures (n_fig) should be an odd number.")

    if file_code is None:
        random_file = random.choice(os.listdir(folderpath))
        file_code = random_file.split('.')[0]
        imgdata = np.load(os.path.join(folderpath, random_file))
    else:
        imgdata = np.load(os.path.join(folderpath, file_code + '.npy'))
    
    
    av_axes = ['x', 'y', 'z']
    if axs not in av_axes:
        print(f"Axis {axs} not valid. Available axes are : x,y,z.")
        sys.exit()
    
    dim = av_axes.index(axs)
    
    train_data = pd.read_csv(train_csv)
    plate_index = train_data.loc[train_data['Image Name'] == file_code, 'Growth Plate Index'].values[0]
    
    # show n_fig' pictures with different x-coordinate that changes by 'step'
    axis_shape = imgdata.shape[dim]
    axx = np.linspace(axis_shape // 2 - step * n_fig // 2, axis_shape // 2 + step * n_fig // 2, num=n_fig)
    
    # create figure
    fig, ax = plt.subplots(1, n_fig, figsize=(n_fig*3, 3))
    fig.suptitle(f"File: {file_code}.npy")
    for i, coor in enumerate(axx):
        if dim == 0:
            ax.flat[i].imshow(imgdata[int(coor), :, :], cmap='bone')
            ax.flat[i].axvline(plate_index, color="red", ls="--")
        if dim == 1:
            ax.flat[i].imshow(imgdata[:, int(coor), :], cmap='bone')
            ax.flat[i].axhline(plate_index, color="red", ls="--")
        if dim == 2:
            ax.flat[i].imshow(imgdata[:, :, int(coor)], cmap='bone')
            
        ax.flat[i].axis('off')
        ax.flat[i].set_title(f"{axs}={int(coor)}")
    plt.tight_layout()
    plt.show()
    

def center_and_crop(width: int = 320,
                    height:int = 330,
                    folderpath: str,
                    outputpath: str):

    # check if output directory exists
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    flist = [os.path.join(folderpath, file) for file in os.listdir(folderpath)]
    for file in tqdm(flist, position=0):
        img_name = file.split('/')[-1].split('.')[0]
        imgarray = load_niifile_as_numpy(file)
        thresh = np.logical_and(imgarray >= 0, imgarray <= 1900).astype(np.uint8)
        for z in range(imgarray.shape[2]):
            cropped_fname = os.path.join(outputpath, f"{img_name}_{z}.npy")
            if os.path.exists(cropped_fname):
                continue
            contours, _ = cv2.findContours(thresh[:,:,z], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                mid_x = imgarray.shape[0] // 2
                mid_y = imgarray.shape[1] // 2
            else:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                mid_x = int(x + w/2)
                mid_y = int(y + h/2)
            
            xmin = int(mid_y - height / 2)
            xmax = int(mid_y + height / 2)
            ymin = int(mid_x - width / 2)
            ymax = int(mid_x + width / 2)
            
            if xmin < 0:
                xmin = 0
                xmax = height
            if ymin < 0:
                ymin = 0
                ymax = width
            if xmax > imgarray.shape[0]:
                xmax = imgarray.shape[0]
                xmin = xmax - height
            if ymax > imgarray.shape[1]:
                ymax = imgarray.shape[1]
                ymin = ymax - width
            
            cropped = imgarray[
                      xmin:xmax,
                      ymin:ymax,
                      z
            ]
            
            cropped = cropped.reshape((cropped.shape[0], cropped.shape[1], 1))
            
            if cropped.shape[0] != height or cropped.shape[1] != width:
                print(f'ERROR: {cropped_fname} shape: {cropped.shape}')
                continue
            np.save(cropped_fname, cropped)