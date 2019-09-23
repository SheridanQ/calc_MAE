# Setup code 
# @author Xiaoxiao Qi
import numpy as np
from numpy import nan
import nibabel as nib
import os, sys
import glob


def split_3_peaks(id_dirpath):
    """
    load 3_direcitons.nii.gz and 3_peaks.nii.gz,
    split them into 3 single peaks
    Output: 2 lists of nibabel objects containing three peaks
    """
    dir3 = os.path.join(id_dirpath, '3_directions.nii.gz')
    pek3 = os.path.join(id_dirpath, '3_peaks.nii.gz')

    # Load nifti file and check format
    dir3_img = nib.load(dir3)
    pek3_img = nib.load(pek3)
    if dir3_img.shape[3] != 9:
        raise ValueError("Wrong format for direction file!")
    if pek3_img.shape[3] != 3:
        raise ValueError("Wrong format for peak file!")

    # split data
    dir3_img_data = dir3_img.get_data()
    pek3_img_data = pek3_img.get_data()

    pdir1 = nib.Nifti1Image(dir3_img_data[:, :, :, 0:3], header=dir3_img.header, affine=dir3_img.affine)
    pdir2 = nib.Nifti1Image(dir3_img_data[:, :, :, 3:6], header=dir3_img.header, affine=dir3_img.affine)
    pdir3 = nib.Nifti1Image(dir3_img_data[:, :, :, 6:9], header=dir3_img.header, affine=dir3_img.affine)
    pp1 = nib.Nifti1Image(pek3_img_data[:, :, :, 0], header=pek3_img.header, affine=pek3_img.affine)
    pp2 = nib.Nifti1Image(pek3_img_data[:, :, :, 1], header=pek3_img.header, affine=pek3_img.affine)
    pp3 = nib.Nifti1Image(pek3_img_data[:, :, :, 2], header=pek3_img.header, affine=pek3_img.affine)

    pdirlist = [pdir1, pdir2, pdir3]
    pplist = [pp1, pp2, pp3]
    # return list(pdir1,pdir2,pdir3), list(pp1,pp2,pp3)
    return pdirlist, pplist


# This function is within an voxel level, and calculate only one peak direction
def calc_v_angles(v1, v2):
    """
    To calculate the angle between two direction vectors.
    Inputs: dir1, dir2 are nibabel objects containing x, y, z direction vectors.
    Output: an 1-D array of angle numbers in float.
    """
    v1=np.array(v1)
    v2=np.array(v2)
    # function to calculate the angle degrees one by one
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'"""
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def angle_calc(v1, v2):
        if np.isnan(v1).any() or not np.any(v2):  # v1==[nan nan nan] or [0 0 0]
            if np.isnan(v2).any() or not np.any(v2):  # v2==[0 0 0] or [nan nan nan]
                angle = 0  ## using zero instead of nan when the corresponding vectors are nans
            else:
                angle = 90
        else:
            if np.isnan(v2).any() or not np.any(v2):
                angle = 90
            else:
                angle = angle_between(v1, v2)
                # if angle > 90??
                if angle > 90:
                    angle = 180 - angle
        return angle

    angle = angle_calc(v1, v2)

    return angle


# This function is within an voxel level, and calculate all peak directions
def calc_v_list_angles(vlist1, vlist2):
    """
    Input: vlist1=[[v1],[v2],[v3],...] vlist2=[[v1],[v2],[v3],...]
    Output: a list with the same length as the vlists that contains the angles between the two vlists.
    """
    if len(vlist1) != len(vlist2):
        raise ValueError('Two vector lists are not the same length! Failed to compare them.')
        return

    num = len(vlist1)
    list_angles = []
    for i in range(num):
        list_angles.append(calc_v_angles(vlist1[i], vlist2[i]))
          # length: num

    min_angle = np.nanmin(np.array(list_angles).astype(np.float64), axis=0)  # scalar

    return min_angle


# This function is for the whole brain voxels
def get_min_angles(dir_list1, dir_list2):
    """
    To get the min angle for each peak after compared with peak from the other subject
    Input: 2 lists of dir files from 2 subject
    """
    if len(dir_list1) != len(dir_list2):
        raise ValueError('Two peak lists are not the same length! Fail to compare them.')
        return

        # make sure the two directions are from the same group of normalized data
    if dir_list1[0].affine.any() != dir_list2[0].affine.any() or dir_list1[0].shape != dir_list2[0].shape:
        raise ValueError("The two direction vectors are not comparable!")
        return

    data1 = [x.get_data() for x in dir_list1]
    data2 = [x.get_data() for x in dir_list2]

    arr_list1 = [x.reshape(-1, x.shape[-1]) for x in data1]
    arr_list2 = [x.reshape(-1, x.shape[-1]) for x in data2]

    # iterate through all voxels
    voxelnum = arr_list1[0].shape[0]
    
    angles = np.zeros(voxelnum)
    for i in range(voxelnum):
        vlist1 = [x[i] for x in arr_list1]
        vlist2 = [x[i] for x in arr_list2]
        # if for both of the images, the voxels are empty
        if np.isnan(vlist1[0]).any() and np.isnan(vlist2[0]).any():
            angles[i]=np.nan
        else:
            angles[i]=calc_v_list_angles(vlist1, vlist2)
        
    # get the mean for the whole brain (igoring voxels outside both images)
    angles_mean = np.nanmean(np.array(angles))
    angles_nonzeromean = np.nanmean(np.where(np.array(angles)!=0, np.array(angles), np.nan))

    return angles_mean, angles_nonzeromean, len(angles), np.count_nonzero(~np.isnan(angles)), np.count_nonzero(angles == 0), np.count_nonzero(angles == 90)

def load_pair_dir(pair_dir1_path, pair_dir2_path):
    dir_list1, pp1 = split_3_peaks(pair_dir1_path)
    dir_list2, pp2 = split_3_peaks(pair_dir2_path)

    return dir_list1, dir_list2


def write_MAE(value, pathname):
    f = open(pathname, 'w')
    f.write(str(value) + '\n')
    f.close()


def main():
    """
	Input: scriptname.py path1 path2 outdir outname
	"""
    pair_dir1_path = str(sys.argv[1])
    pair_dir2_path = str(sys.argv[2])
    dir_list1, dir_list2 = load_pair_dir(pair_dir1_path, pair_dir2_path)

    MAE = get_min_angles(dir_list1, dir_list2)
    print(MAE)
    write_MAE(str(MAE[0])+','+str(MAE[1])+','+str(MAE[2])+','+str(MAE[3])+','+str(MAE[4]), sys.argv[3])


if __name__=="__main__":
 	main()
