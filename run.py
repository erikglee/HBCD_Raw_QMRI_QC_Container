#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil, gzip
import dipy
import warnings
import nibabel as nib
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from dipy import align
import dipy
import argparse
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D)


def replace_file_with_gzipped_version(file_path):
    '''Replace a file with a gzipped version of itself
    
    Parameters
    ----------
    file_path : str
        Path to the file to be gzipped
    
    '''
    
    #file_path = path to the file to be gzipped
    
    with open(file_path, 'rb') as f_in:
        with gzip.open(file_path + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
    os.remove(file_path)
    
    return file_path + '.gz'

def calc_synth_t1w_t2w(t1map_path, t2map_path, pdmap_path, output_folder, subject_name, session_name):
    
    print('   Calculating synthetic T1w and T2w images from QALAS maps')
    t1_tr = 10*1000
    t1_te = 0.00224*1000

    t2_tr = 10*1000
    t2_te = 0.1*1000

    temp_t1_img = nib.load(t1map_path)
    temp_t1_data = temp_t1_img.get_fdata()
    temp_t2_data = nib.load(t2map_path).get_fdata()
    temp_pd_data = nib.load(pdmap_path).get_fdata()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t1w = temp_pd_data*(1.0 - np.exp(-t1_tr/temp_t1_data))*np.exp(-t1_te/temp_t2_data)
        t2w = temp_pd_data*(1.0 - np.exp(-t2_tr/temp_t1_data))*np.exp(-t2_te/temp_t2_data)

    #Output Destination = the top directory for outputs (i.e. same for all subjects)    
    input_path_split = t1map_path.split('/')
    if 'ses-' in input_path_split[-3]:
        partial_sub_path = '/'.join(input_path_split[-4:])
        beginning_path = '/'.join(input_path_split[:-4])
    else:
        partial_sub_path = '/'.join(input_path_split[-3:])
        beginning_path = '/'.join(input_path_split[:-3])

    anat_out_dir = os.path.join(output_folder, os.path.dirname(partial_sub_path), 'qalas_derived_weighted_images')
    if os.path.exists(anat_out_dir) == False:
        os.makedirs(anat_out_dir)
        
    t1w_nifti_img = nib.nifti1.Nifti1Image(t1w, affine = temp_t1_img.affine, header = temp_t1_img.header)
    t2w_nifti_img = nib.nifti1.Nifti1Image(t2w, affine = temp_t1_img.affine, header = temp_t1_img.header)
    
    t1w_name = os.path.join(anat_out_dir, '{}_{}_space-QALAS_desc-synthetic_T1w.nii.gz'.format(subject_name, session_name))
    t2w_name = os.path.join(anat_out_dir, '{}_{}_space-QALAS_desc-synthetic_T2w.nii.gz'.format(subject_name, session_name))

    nib.save(t1w_nifti_img, t1w_name)
    nib.save(t2w_nifti_img, t2w_name)
    
    return t1w_name, t2w_name

def register_images(input_file_path, output_file_path,
                    additional_images = None):
    '''
    Parameters
    ----------

    input_file_path : str
        Path to nifti file with T1w or T2w BIDS naming that
        will be registered to MNI image.
    output_file_path : str
        Path to the registered file that will be created if 
        input_file_path has T1w.nii.gz or T2w.nii.gz, the
        output file must also have this.
    additional_images : dict
        Additional images that are already registered with the
        input_file that should be registered using the same
        transformation. Keys represent input file paths, and
        values should represent the new file being created.

    Returns
    -------

    Path to a generic binary brain mask file created from the input
    file. This path is used for downstream visualization tasks
    
    '''
    

    anat_out_dir = '/'.join(output_file_path.split('/')[-1])
    if os.path.exists(anat_out_dir) == False:
        os.makedirs(anat_out_dir)
        
    if 'T1w' in input_file_path:
        contrast = 'T1w'
        stripped_out_file = os.path.join(output_file_path).replace('T1w.nii', 'masked-brain_T1w.nii')
        generic_out_mask = os.path.join(output_file_path).replace('T1w.nii', 'masked-brain.nii')
    elif 'T2w' in input_file_path:
        contrast = 'T2w'
        stripped_out_file = os.path.join(output_file_path).replace('T2w.nii', 'masked-brain_T2w.nii')
        generic_out_mask = os.path.join(output_file_path).replace('T2w.nii', 'masked-brain.nii')
        
    if os.path.exists(output_file_path):
        print('Using already existing registered out file with name: {}'.format(output_file_path))
        return generic_out_mask
    
    os.system('python3 /freesurfer/mri_synthstrip -i {input_path} -o {output_skull_stripped_path} -m {generic_out_mask}'.format(
        input_path = input_file_path, output_skull_stripped_path = stripped_out_file, generic_out_mask=generic_out_mask))
    

    ###NEED TO UPDATE THIS STUFF FOR ADDITIONAL T1/T2/PD maps to be saved

    print('Attempting Native to MNI Infant Registration using DIPY: ')
    template_image_path = '/image_templates/tpl-MNIInfant_cohort-1_res-1_mask-applied_{}.nii.gz'.format(contrast)
    template_image = dipy.io.image.load_nifti(template_image_path)
    original_image = dipy.io.image.load_nifti(input_file_path)
    registered_img = align.affine_registration(stripped_out_file, template_image[0], static_affine=template_image[1])
    
    #This returns a list with the following elements:
    #transformed : array with moving data resampled to the static space
    #after computing the affine transformation
    #affine : the affine 4x4 associated with the transformation.
    #xopt : the value of the optimized coefficients.
    #fopt : the value of the optimization quality metric.
    
    #(registered out path is the name of the skull stripped image that has
    #been registered)
    registered_out_path = stripped_out_file.replace('{}.nii'.format(contrast), 'reg-MNIInfant_{}.nii'.format(contrast))
    dipy.io.image.save_nifti(registered_out_path,
                         registered_img[0], template_image[1])
    
    #This is now applying the registration to the full image without skull stripping
    affine_map = AffineMap(registered_img[1],
                       original_image[0].shape, original_image[1],
                       original_image[0].shape, original_image[1])
    resampled = affine_map.transform(original_image[0])
    dipy.io.image.save_nifti(output_file_path,
                     resampled, original_image[1])
    
    #Iterate through all additional images defined by keys,
    #and apply registration, saving images to the locations
    #under values
    if type(additional_images) != type(None):
        for temp_img in additional_images.keys():
                
                original_image = dipy.io.image.load_nifti(temp_img)

                affine_map = AffineMap(registered_img[1],
                       original_image[0].shape, original_image[1],
                       original_image[0].shape, original_image[1])
                resampled = affine_map.transform(original_image[0])
                dipy.io.image.save_nifti(additional_images[temp_img],
                                resampled, original_image[1])

    return generic_out_mask

def make_slices_image(image_nifti_path, slice_info_dict, output_img_name, close_plot = True,
                     upsample_factor = 2, mask_path = None, vmin_multiplier = 0.3,
                     vmax_multiplier = 1.7):
    '''Takes a nifti and plots slices of the nifti according to slices_info_dict
    
    Parameters
    ----------
    image_nifti_path : str
        Path to nifti image to make plot with
    slice_info_dict : dict
        Dictionary that formats how the picture
        will be formatted. See example below.
    output_img_name : str
        The name/full path of the image
        to be created by this function
    close_plot : bool, default True
        Whether to close the plot after it
        is rendered
    mask_path : str or None, default None
        A binary brain mask.
        This will can be used to help with image contrast.
    vmin_multiplier : float, default 
        
    Example slice_info_dict. The first entry in each key's
    list dictates which plane is being imaged. The second
    entry indicates where (in RAS) the center of the plane
    should be placed. And the third and fourth entries dictate
    the range of voxels to be included in the slice. For example,
    100 would mean that 200 units are included. Larger values
    will make larger field of views. 
    
    slice_info_dict = {'coronal_1' : [0, -25, 100, 100],
                   'coronal_2' : [0, 0, 100, 100],
                   'coronal_3' : [0, 25, 100, 100],
                   'sagittal_1' : [1, -50, 100, 100],
                   'sagittal_2' : [1, 0, 100, 100],
                   'sagittal_3' : [1, 30, 100, 100],
                   'axial_1' : [2, -50, 100, 100],
                   'axial_2' : [2, 0, 100, 100],
                   'axial_3' : [2, 50, 100, 100]}
    
    '''
    
    #Load the nifti image
    nifti_image = nib.load(image_nifti_path)

    #Grab data + affine
    full_data = nifti_image.get_fdata()
    full_affine = nifti_image.affine
    
    #Load nifti mask (assume voxels > 0.5 are good)
    if type(None) == type(mask_path):
        vmin = None
        vmax = None
    else:
        mask_data = nib.load(mask_path).get_fdata()
        mask_data = mask_data*full_data
        mask_vals = mask_data[mask_data > 0.5]
        #vmin = np.percentile(mask_vals, 1)
        #vmax = np.percentile(mask_vals, 95)
        hist_results = np.histogram(mask_vals, bins = 100)
        modal_value = hist_results[1][np.argmax(hist_results[0])]
        vmin = modal_value*vmin_multiplier
        vmax = modal_value*vmax_multiplier
    
    
    #Setup interpolator in scipy so we can
    #resample the image in RAS units instead
    #of voxel units
    i = np.arange(0,full_data.shape[0])
    j = np.arange(0,full_data.shape[1])
    k = np.arange(0,full_data.shape[2])
    interp = RegularGridInterpolator((i, j, k), full_data, method = 'linear', bounds_error = False)
    
    inv_affine = np.linalg.inv(full_affine) #To get to RAS
    imgs = [] #List to store all of the individual slice pixel intensities
    
    #Make each of the slice images
    for temp_img in slice_info_dict.keys():
        temp_setup = slice_info_dict[temp_img]
        temp_slice = []
        
        #Upsample by a factor of 2
        for i in range(-1*temp_setup[2]*upsample_factor,temp_setup[2]*upsample_factor):
            for j in range(-1*temp_setup[3]*upsample_factor, temp_setup[3]*upsample_factor):
                i_hat = i/upsample_factor
                j_hat = j/upsample_factor
                if temp_setup[0] == 0:
                    temp_slice.append(np.matmul(inv_affine, np.array([temp_setup[1],i_hat,j_hat,1])))
                elif temp_setup[0] == 1:
                    temp_slice.append(np.matmul(inv_affine, np.array([i_hat,temp_setup[1],j_hat, 1])))
                elif temp_setup[0] == 2:
                    temp_slice.append(np.matmul(inv_affine, np.array([i_hat,j_hat,temp_setup[1],1])))
                else:
                    raise ValueError('Error: the second entry must be 0,1,2 to indicate slicing axis')
        vals = interp(np.array(temp_slice)[:,0:3])
        imgs.append(np.rot90(vals.reshape((temp_setup[2]*2*upsample_factor, temp_setup[3]*2*upsample_factor))))

    dim1 = imgs[0].shape[0]
    dim2 = imgs[0].shape[1]
    full_img_panel = np.zeros((dim1*3, dim2*3))
    for i, temp_img in enumerate(imgs):
        y = np.mod(i, 3)
        x = np.floor(i/3)
        full_img_panel[int(x*dim1):int((1+x)*dim1),int(y*dim2):int((1+y)*dim2)] = temp_img
    full_img_panel[np.where(np.isnan(full_img_panel))] = 0


    fig = plt.figure(dpi = 250)
    plt.imshow(full_img_panel, cmap = 'gist_gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(output_img_name, bbox_inches='tight', pad_inches = 0)
    if close_plot:
        plt.close()
    
    return


#Configure the commands that can be fed to the command line
parser = argparse.ArgumentParser()
parser.add_argument("bids_dir", help="The path to the BIDS directory for your study (this is the same for all subjects)", type=str)
parser.add_argument("output_dir", help="The path to the folder where outputs will be stored (this is the same for all subjects)", type=str)
parser.add_argument("analysis_level", help="Should always be participant", type=str)

parser.add_argument('--participant_label', '--participant-label', help="The name/label of the subject to be processed (i.e. sub-01 or 01)", type=str)
parser.add_argument('--session_id', '--session-id', help="OPTIONAL: the name of a specific session to be processed (i.e. ses-01)", type=str)
parser.add_argument('--matplotlib_contrast', '--matplotlib-contrast', help="Use matplotlib to determine image contrast instead of brain mask intensity statistics.", action='store_true')
args = parser.parse_args()


#Get cwd in case relative paths are given
cwd = os.getcwd()

#reassign variables to command line input
bids_dir = args.bids_dir
if os.path.isabs(bids_dir) == False:
    bids_dir = os.path.join(cwd, bids_dir)
output_dir = args.output_dir
if os.path.isabs(output_dir) == False:
    output_dir = os.path.join(cwd, output_dir)
analysis_level = args.analysis_level
if analysis_level != 'participant':
    raise ValueError('Error: analysis level must be participant, but program received: ' + analysis_level)
matplotlib_contrast = args.matplotlib_contrast


#Set session label
if args.session_id:
    session_label = args.session_id
    if 'ses-' not in session_label:
        session_label = 'ses-' + session_label
else:
    session_label = None
    
#Find participants to try running
if args.participant_label:
    participant_split = args.participant_label.split(' ')
    participants = []
    for temp_participant in participant_split:
        if 'sub-' not in temp_participant:
            participants.append('sub-' + temp_participant)
        else:
            participants.append(temp_participant)
else:
    os.chdir(bids_dir)
    participants = glob.glob('sub-*')
    
#Dictionary for making slices
slice_info_dict = {'coronal_1' : [0, -25, 100, 100],
                   'coronal_2' : [0, 0, 100, 100],
                   'coronal_3' : [0, 25, 100, 100],
                   'sagittal_1' : [1, -50, 100, 100],
                   'sagittal_2' : [1, 0, 100, 100],
                   'sagittal_3' : [1, 30, 100, 100],
                   'axial_1' : [2, -50, 100, 100],
                   'axial_2' : [2, 0, 100, 100],
                   'axial_3' : [2, 50, 100, 100]}
    
#Iterate through all participants
for temp_participant in participants:
    
    #Check that participant exists at expected path
    subject_path = os.path.join(bids_dir, temp_participant)
    if os.path.exists(subject_path):
        os.chdir(subject_path)
    else:
        raise AttributeError('Error: no directory found at: ' + subject_path)
    
    #Find session/sessions
    if session_label == None:
        sessions = glob.glob('ses*')
        if len(sessions) < 1:
            sessions = ['']
    elif os.path.exists(session_label):
        sessions = [session_label]
    else:
        raise AttributeError('Error: session with name ' + session_label + ' does not exist at ' + subject_path)

    #Iterate through sessions
    for temp_session in sessions:

        #If there is no session structure, this will go to the subject path
        session_path = os.path.join(subject_path, temp_session)
        
        #Grab T2w file
        anats_dict = {'T2_images' : [],
                      'T1_images' : [],
                      'PD_images' : []}
        t2_anats = glob.glob(os.path.join(session_path,'anat/*T2map.ni*'))

        for temp_t2 in t2_anats:
            t1 = temp_t2.replace('T2map.nii', 'T1map.nii')
            pd = temp_t2.replace('T2map.nii', 'PDmap.nii')
            if os.path.exists(t1) and os.path.exists(pd):
                anats_dict['T2_images'].append(temp_t2)
                anats_dict['T1_images'].append(t1)
                anats_dict['PD_images'].append(pd)

        
          

            
        for i, temp_t2 in enumerate(anats_dict['T2_images']):

            #Make output anat folder for subject/session if it doesnt exist
            out_anat_folder = os.path.join(output_dir, temp_participant, temp_session, 'anat')
            if os.path.exists(out_anat_folder) == False:
                os.makedirs(out_anat_folder)

            #First create a synthetic T2w image for registration
            t1w_path, t2w_path = calc_synth_t1w_t2w(anats_dict['T1_images'][i], anats_dict['T2_images'][i], anats_dict['PD_images'][i], output_dir, temp_participant, temp_session)

            #Register synthetic t2w image to the MNI template
            registered_t2w_name = os.path.join(out_anat_folder, t2w_path.split('/')[-1]).replace('T2w.nii.gz', 'reg-MNIInfant_T2w.nii')
            registered_t1_name = registered_t2w_name.replace('T2w.nii', 'T1map.nii').replace('_space-QALAS', '')
            registered_t2_name = registered_t2w_name.replace('T2w.nii', 'T2map.nii').replace('_space-QALAS', '')
            registered_pd_name = registered_t2w_name.replace('T2w.nii', 'PDmap.nii').replace('_space-QALAS', '')
            generic_mask_path = register_images(t2w_path,
                                                registered_t2w_name,
                                                additional_images = {anats_dict['T1_images'][i] : registered_t1_name,
                                                                        anats_dict['T2_images'][i] : registered_t2_name,
                                                                        anats_dict['PD_images'][i] : registered_pd_name})
            os.remove(registered_t2w_name) #dont actually need this
            os.remove(registered_t2w_name.replace('_reg-MNIInfant', '_reg-MNIInfant_masked-brain')) #dont need this either
            vmin_multipliers = [0.5, 0.3, 0.5]
            vmax_multipliers = [1.5, 1.7, 1.5]
            
            for j, temp_reg in enumerate([registered_t1_name, registered_t2_name, registered_pd_name]):
                
                slice_img_path = temp_reg.replace('.nii', '_image-slice.png')
                slice_img_path = slice_img_path.replace('slice.png.gz', 'slice.png') #For case when nifti is compressed
                if matplotlib_contrast == False:
                    make_slices_image(temp_reg, slice_info_dict, slice_img_path, close_plot = True,
                            upsample_factor = 2, mask_path = generic_mask_path,
                            vmin_multiplier=vmin_multipliers[j], vmax_multiplier=vmax_multipliers[j])
                else:
                    make_slices_image(temp_reg, slice_info_dict, slice_img_path, close_plot = True,
                            upsample_factor = 2)
                    
            for temp_uncompressed in [registered_t1_name, registered_t2_name, registered_pd_name, generic_mask_path]:
                replace_file_with_gzipped_version(temp_uncompressed)
                
            #Delete the synthetic image folder
            synthetic_image_folder = '/'.join(t1w_path.split('/')[:-1])
            shutil.rmtree(synthetic_image_folder)


        print('Finished with: {}'.format(session_path))