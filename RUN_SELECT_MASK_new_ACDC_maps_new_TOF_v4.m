% 
%   ASL optimization software
% 
%    METHOD 2 (used when qform_code > 0, which should be the "normal" case):
%    ---------------------------------------------------------------------
%    The (x,y,z) coordinates are given by the pixdim[] scales, a rotation
%    matrix, and a shift.  This method is intended to represent
%    "scanner-anatomical" coordinates, which are often embedded in the
%    image header (e.g., DICOM fields (0020,0032), (0020,0037), (0028,0030),
%    and (0018,0050)), and represent the nominal orientation and location of
%    the data.  This method can also be used to represent "aligned"
%    coordinates, which would typically result from some post-acquisition
%    alignment of the volume to a standard orientation (e.g., the same
%    subject on another day, or a rigid rotation to true anatomical
%    orientation from the tilted position of the subject in the scanner).
%    The formula for (x,y,z) in terms of header parameters and (i,j,k) is:
% 
%      [ x ]   [ R11 R12 R13 ] [        pixdim[1] * i ]   [ qoffset_x ]
%      [ y ] = [ R21 R22 R23 ] [        pixdim[2]  j ] + [ qoffset_y ]
%      [ z ]   [ R31 R32 R33 ] [ qfac  pixdim[3] * k ]   [ qoffset_z ]
% 
% 
%    The qoffset_* shifts are in the NIFTI-1 header.  Note that the center
%    of the (i,j,k)=(0,0,0) voxel (first value in the dataset array) is
%    just (x,y,z)=(qoffset_x,qoffset_y,qoffset_z).
% 
% 
%    The rotation matrix R is calculated from the quatern_* parameters.
%    This calculation is described below.
% 
% 
%    The scaling factor qfac is either 1 or -1.  The rotation matrix R
%    defined by the quaternion parameters is "proper" (has determinant 1).
%    This may not fit the needs of the data; for example, if the image
%    grid is
%      i increases from Left-to-Right
%      j increases from Anterior-to-Posterior
%      k increases from Inferior-to-Superior
%    Then (i,j,k) is a left-handed triple.  In this example, if qfac=1,
%    the R matrix would have to be
% 
%      [  1   0   0 ]
%      [  0  -1   0 ]  which is "improper" (determinant = -1).
%      [  0   0   1 ]
% 
%
% Version notes:
% 
%
% NEED TO FIX LEFT-RIGHT BUG!!!!
%
%
% Scan notes:  Subject 3T 13:  Labeling plane slice in raw TOF: 34


% don't use NATURAL method for scatteredInterpolant


% STEPS -------------------------------------------------------------------
%
% 1.  Put TOF in ./dicom/TOF
% 2.  Put magnitude from B0 map into ./dicom/mag
% 3.  Put phase from B0 map into ./dicom/fieldmap
% 4.  Make sure coil basis field maps and DICOM reference are in specified
% folders
%

clear all,% close all


addpath('./nifti/')
delete('./nifti/*')
% addpath('/Users/stockmann/mri/mgh/shim_array/asl/experiments/ASL_process/forLincoln_shimfun/'); % this folder has overlapping names with matlab native scripts and caused weird errors when added to path

% need to specify path with ASL solver code
 rmpath('/Users/stockmann/mri/mgh/shim_array/asl/experiments/ASL_process/shimfun_reduced_inventory');
% addpath('/Users/stockmann/jason_dropbox/Coil_designs_JS_LCB/32-42_AC-DC_GE/LCB_archive/LCB_Projects/ASL_workflow_old/ASL_process/')
% -------------------------------------------------------------------------
which_coil_to_use = 2;  % 1 for 12ch neck coil; 0 for 32ch 3T head coil; 2 for 32ch 7T coil
use_lincoln_sim_coil = 0; % might be broken
downsample_switch = 0;  % save processing time by downsampling matrices --> BROKEN
% -------------------------------------------------------------------------

% load lincoln custom neck coil
%  load /Users/stockmann/mri/mgh/shim_array/asl/simulations/neck_coil_sim_v2_for_abstract/neck_coils_12ch.mat

 
% delete these lines
% load /Users/stockmann/mgh_data/shim_data/skyra_data/prisma_shimcoil_field_mapping_multi_gre_sept_3_2019/prisma_field_maps_75sli_3T_shim_coil_silicon_oil_four_echo_GRE_WIENER_FILTER.mat
% coils=coil.fm;

% mri_convert_path = '/Applications/freesurfer/bin/mri_convert'
% setenv('FREESURFER_HOME','/Applications/freesurfer')

dTE = 0.00246;

% INSTRUCTIONS:  COIL BASIS IS AUTOMATICALLY LOADED HERE.  PUT TOF IN DICOM
% SUBFOLDER AND NAME IT TOF.  PUT BASELINE FIELD MAP IN SUBFOLDER AS WELL.
% new 75 slice field maps --- CHANGE THESE PATHS
if which_coil_to_use == 0
%     load('/Users/stockmann/mgh_data/shim_data/skyra_data/new_prisma_coil_basis_automated_better_accuracy_oct_20_2022/bay4_cal_offsetCorrected_10_20.mat')
    % extended neck coverage
    load('/Users/stockmann/mgh_data/shim_data/skyra_data/automated_field_mapping_shimarray_extend_to_neck_coverage_sept_26_2023/coil_32ch_acdc_EXTENDED_NECK_sept_25_2023.mat');
    save /Users/stockmann/jason_dropbox/ASL_materials_lincoln_jason_yulin_meher/figures_code/mat/head_coils.mat coil
    coil_basis_mat = './mat/head_coils.mat'; % mat file for the coils
%     coil_basis_info_dicom_path = '/Users/stockmann/mgh_data/shim_data/skyra_data/new_prisma_coil_basis_automated_better_accuracy_oct_20_2022/dicoms/GRE_FM_BASE_0002'; % folder for dicom series for metadata
    
    % new extended neck coverage basis acquired sept 2023
    coil_basis_info_dicom_path = '/Users/stockmann/mgh_data/shim_data/skyra_data/automated_field_mapping_shimarray_extend_to_neck_coverage_sept_26_2023/dicoms/GRE_FM_BASE_0022';
elseif which_coil_to_use == 1 % 12ch coil case
    coil_basis_mat = './mat/head_coils_12ch.mat'; % mat file for the coils
    coil_basis_info_dicom_path = '/Users/stockmann/jason_dropbox/ASL_materials_lincoln_jason_yulin_meher/scan_data/coil_field_mapping_basis_march_23_2021/FM_C21_P400_0047'; % folder for dicom series for metadata for 12ch coil
elseif which_coil_to_use == 2 % 7T coil case
    load('/Users/stockmann/mgh_data/shim_data/7T_shim_coil/field_maps_acquired_with_arnaud_for_ShimToolbox_Feb_2024_low_SNR_extended_coverage/coil_32ch_acdc_EXTENDED_NECK_arnaud_jan_2024.mat');
    save /Users/stockmann/jason_dropbox/ASL_materials_lincoln_jason_yulin_meher/figures_code/mat/head_coils.mat coil
    coil_basis_mat = './mat/head_coils.mat'; % mat file for the coils
    coil_basis_info_dicom_path = '/Users/stockmann/mgh_data/shim_data/7T_shim_coil/field_maps_acquired_with_arnaud_for_ShimToolbox_Feb_2024_low_SNR_extended_coverage//./GRE_FM_BASELINE_0017/MAP_SHIM_COIL_PROFILES.MR.WALD_JASON.0017.0090.2024.01.31.23.10.48.108068.854532392.IMA';
end
%% THESE ARE THE INPUT DICOMS NEEDED FOR EACH SUBJECT!!!!!!!!!!!!!!!!!!!!!!
tof_dicom_path = './dicom/TOF'; % path to TOF dicom
fieldmap_dicom_path = './dicom/fieldmap'; % path to fieldmap dicom
mag_dicom_path = './dicom/mag'; % path to fieldmap dicom
%% ========================================================================


clear_nifti = true;
 
% =========================================================================
load(coil_basis_mat)


if clear_nifti
    delete('./nifti/*');
end

dcm2niix_wrapper(coil_basis_info_dicom_path, './nifti',['/original_coil']);
temp=load_nifti('./nifti/original_coil_e2.nii');  temp.vol=rot90((coil.fm),-1);  
save_nifti(temp,'./nifti/starting_coil_basis.nii');
dcm2niix_wrapper(tof_dicom_path, './nifti','tof');
dcm2niix_wrapper(fieldmap_dicom_path, './nifti','starting_fmap');
dcm2niix_wrapper(mag_dicom_path, './nifti','mag');


tic

% LOAD MAGNITUDE -----------------------------
% commented out 5/22/2024 JPS
% eval(['unix(''flirt -in ./nifti/','mag_e2.nii',' -ref ./nifti/','tof.nii',' -omat ./nifti/invol2refvol.mat -interp nearestneighbour -applyxfm -usesqform -out ./nifti/mag.nii -v'')']);

eval(['unix(''flirt -in ./nifti/','mag_e1.nii',' -ref ./nifti/','tof.nii',' -omat ./nifti/invol2refvol.mat -interp nearestneighbour -applyxfm -usesqform -out ./nifti/mag.nii -v'')']);

temp2=load_nifti('./nifti/mag.nii.gz');
mag=fix_nifti(temp2);
[mag_mask temptemp] = simple_magnitude_threshold(mag,.9);
tic


disp('interpolating COIL BASIS into TOF coordinate system, may take some time...............'),
eval(['unix(''flirt -in ./nifti/','starting_coil_basis.nii',' -ref ./nifti/','tof.nii',' -omat ./nifti/invol2refvol.mat -interp nearestneighbour -applyxfm -usesqform -out ./nifti/coil_interp_out.nii -v'')']);
toc
temp2=load_nifti('./nifti/coil_interp_out.nii.gz');
coil_interp=fix_nifti(temp2);
toc

% LOAD PHASE / FMAP -----------------------------
eval(['unix(''flirt -in ./nifti/','starting_fmap_e2_ph.nii',' -ref ./nifti/','tof.nii',' -omat ./nifti/invol2refvol.mat -interp nearestneighbour -applyxfm -usesqform -out ./nifti/fmap_interp_out.nii -v'')']);
% commented out 5/22/2024 JPS
% eval(['unix(''flirt -in ./nifti/','starting_fmap_e2_ph.nii',' -ref ./nifti/','tof.nii',' -omat ./nifti/invol2refvol.mat -interp nearestneighbour -applyxfm -usesqform -out ./nifti/fmap_interp_out.nii -v'')']);

temp3=load_nifti('./nifti/fmap_interp_out.nii.gz');
fmap_interp=fix_nifti(temp3, dTE);

disp('...DONE WITH ALL INTERPOLATION!')


temp4=load_nifti('./nifti/tof.nii');
tof=fix_nifti(temp4);

coil_old = coil.fm;
coil.fm=coil_interp;
save('./mat/tof', 'tof')

% replace old coil basis with simulated -----------------------------------
if use_lincoln_sim_coil
    ns = numel(tof(1,1,:));
    coils2 = permute(coils,[2 1 3 4]);  coil2=flipud(coils2);
    coil.fm = coils2(:,:,13:13+ns-1,:);  % fudge factor for Lincoln's coordinate system
end

% make sure mask is zero outside the coil profiles
mask_coil = ones(size(coil_interp(:,:,:,1)));
for sss=1:numel(coil_interp(1,1,:,1))
    if max(max(max(coil_interp(:,:,sss,1)))) == 0;
        mask_coil(:,:,sss) = 0;
    end
end




%% select vessel mask  -- RE-RUN THIS SECTION TO AVOID INTERPOLATING AGAIN
% -------------------------------------------------------------------------
unix('python3 tof_artery_mask_june_2024.py')
python_temp = load_np('./npy/want_0_start_voxel.npy');

load('./mat/mask.mat');
shim.mask=mask; 
shim = struct('coil', coil, 'fmap', fmap_interp, 'mask', mask);  
sh1+1
im.mask=mask;   % exclude areas where coil profiles don't exist
shim.mask_nan = double(shim.mask);  shim.mask_nan(shim.mask_nan == 0 ) = nan;  
% save('./mat/shim', 'shim');

% -------------------------------------------------------------------------

%%
% figure(10),imagesc([vol2mos(shim.fmap) vol2mos(shim.coil.fm(:,:,:,10)) vol2mos(double(shim.mask*400))],[-25 25]),axis image off

% clear coil



%% DOWNSAMPLE TO SPEED UP PROCESSING  -- NOT USED
 
downsample_factor = 1;
if downsample_switch
   shim.fmap=shim.fmap(1:downsample_factor:end,1:downsample_factor:end,1:2:end); 
   shim.mask = shim.mask(1:downsample_factor:end,1:downsample_factor:end,1:2:end);
   shim.coil.fm = shim.coil.fm(1:downsample_factor:end,1:downsample_factor:end,1:2:end,:);
   shim.mag_mask = mag_mask(1:downsample_factor:end,1:downsample_factor:end,1:2:end,:);
   shim.mag = mag(1:downsample_factor:end,1:downsample_factor:end,1:2:end,:);
   shim.tof = tof(1:downsample_factor:end,1:downsample_factor:end,1:2:end,:);
else 
    shim.mag=mag;
    shim.tof=tof;
    shim.mag_mask=mag_mask;
end
shim.delta_TE=.00246;

% compensate for B0 offset?
compensate_B0 = 0;
clear opts
opts{3} = 200;
opts{4} = 2.9;
opts{5} = 0;
opts{6} = .5; % alpha
% opts{7} = 3000; % highbw
% opts{8} = 1000; % lowbw


max_current = 2;
total_current = 20;
total_HARD_LIMIT_CURRENT = 35;

if which_coil_to_use == 0
    shim.coil.fm(:,:,:,21) = 0;  
end

% % phase unwrapping -- needs debugging
% shim.unshimmed = shim.fmap;
% shim = mrir_phase_unwrap__prelude_standalone(shim);



% hack to zero dead channels on 12ch coil  --> 0-indexed Ch20 which is
% confirmed dead now, but was active when coil field map basis set was acquired
% jps hack 4/13/23
% shim.coil.fm(:,:,:,2:8) = 0;
% shim.coil.fm(:,:,:,11:12) = 0;
% shim.coil.fm(:,:,:,17:19) = 0;


% shim.coil.fm(:,:,:,21) = 0;  % reintroduce this channel because broken
% wiring in shim cables has been fixed JPS 7/19/2023

% shim.coil.fm(:,:,:,25:32) = 0;

%% create alternative mask for homogeneity shim ---------------------------
% need to include downsampling factor here
slice_to_mask = python_temp(3);  
num_slices = 6;

figure(501),imagesc(shim.tof(:,:,slice_to_mask)),colormap('gray'),axis image off
shim.mask_rectangle = zeros(size(shim.tof));
shim.mask_rectangle(:,:,slice_to_mask) = roipoly;
for ss=slice_to_mask-num_slices : slice_to_mask+num_slices
    shim.mask_rectangle(:,:,ss) = shim.mask_rectangle(:,:,slice_to_mask);
end
shim.mask_rectangle = logical(shim.mask_rectangle);
shim.mask_rectangle = shim.mask_rectangle .* logical(shim.mag_mask);
shim.mask_nan_rectangle = double(shim.mask_rectangle);  shim.mask_nan_rectangle(shim.mask_nan_rectangle == 0 ) = nan;

%%  COMPUTE artery mask homogeneity shim ----------------------------------
disp('Computing homogeneity shim over arterial vessel mask...')

disp('VESSEL HOMOGENEITY MASK')
shim.mask_all_ones = logical(shim.mask);
[shim.shimmed_homogeneity,amps,std_unshimmed,std_shimmed] = perform_shim_quad(shim.fmap,shim.mask_all_ones,shim.coil.fm,2,20);

% display shim coefficients on screen
disp(['TOTAL CURRENT VESSEL MASK: ',num2str((sum(abs(amps))))])
for tt=1:numel(shim.coil.fm(1,1,1,:)),disp([num2str(round(1000*amps(tt))/1000),', ']),end


% shim over a rectangular mask
disp('Computing homogeneity shim over RECTANGULAR mask...')

disp('RECTANGULAR MASK')
[shim.shimmed_homogeneity_rect,amps_rect,std_unshimmed_rect,std_shimmed_rect] = perform_shim_quad(shim.fmap,shim.mask_rectangle,shim.coil.fm,2,20);

% display shim coefficients on screen
disp('')
disp(['TOTAL CURRENT RECTANGULAR MASK: ',num2str((sum(abs(amps_rect))))])

for tt=1:numel(shim.coil.fm(1,1,1,:)),disp([num2str(amps_rect(tt)),', ']),end




% use amps or amps_rect for this section
shim.coil_shim_field_everywhere =  apply_current_using_coil_basis(amps, shim.coil.fm);

shim.coil_shim_field_everywhere_rect =  apply_current_using_coil_basis(amps_rect, shim.coil.fm);



figure(301),imagesc([vol2mos(shim.fmap) vol2mos(shim.coil_shim_field_everywhere + shim.fmap)],[-100 100]),colormap('jet'),colorbar,axis image off
title('VESSEL baseline field map on left, applied total field on right')

figure(302),imagesc([vol2mos(shim.fmap) vol2mos(shim.coil_shim_field_everywhere_rect + shim.fmap)],[-100 100]),colormap('jet'),colorbar,axis image off
title('RECT baseline field map on left, applied total field on right')


figure(303),imagesc2([vol2mos(double(shim.mask_nan(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices)).*shim.fmap(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices)) vol2mos(shim.mask_nan(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices).*shim.shimmed_homogeneity(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices))  ],[-100 100]),colormap('jet'),colorbar,axis image off
title('VESSEL: baseline field map on left, applied total field on right')

figure(304),imagesc2([vol2mos(double(shim.mask_nan_rectangle(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices)).*shim.fmap(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices)) vol2mos(shim.mask_nan_rectangle(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices).*shim.shimmed_homogeneity_rect(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices))  ],[-100 100]),colormap('jet'),colorbar,axis image off
title('RECT: baseline field map on left, applied total field on right')
add_mask_outline([vol2mos(shim.mask(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices)) vol2mos(shim.mask(:,:,slice_to_mask-num_slices:slice_to_mask+num_slices)) ])








return

%% shim whole slice for homogeneity
% [shim.shimmed_homogeneity,amps_whole_slice,std_unshimmed,std_shimmed] = perform_shim_quad(shim.fmap,shim.mag_mask,shim.coil.fm,2.5,25);
% disp(['TOTAL CURRENT WHOLE SLICE MASK: ',num2str((sum(abs(amps_whole_slice))))])
% for tt=1:numel(shim.coil.fm(1,1,1,:)),disp([num2str(amps(tt)),', ']),end
% shim.coil_shim_field_everywhere =  apply_current_using_coil_basis(amps, shim.coil.fm);
% 
% 
% 
% figure(302),imagesc([vol2mos(shim.fmap) vol2mos(shim.coil_shim_field_everywhere + shim.fmap)],[-100 100]),colormap('jet'),colorbar,axis image off
% title('baseline field map on left, applied total field on right WHOLE SLICE')
% 
% 







%%
% --------------------------------------------------------------------
%  Nick's solver for territory mapping
% --------------------------------------------------------------------

% addpath('/Users/stockmann/mri/mgh/shim_array/asl/experiments/ASL_process/shimfun_reduced_inventory');

% rmpath('/Users/stockmann/jason_dropbox/ASL_materials_lincoln_jason_yulin_meher/figures_code/shimfun_only_necessary_functions');

% rmpath('/Users/stockmann/Dropbox (Partners HealthCare)/ASL_materials_lincoln_jason_yulin_meher/figures_code/forLincoln_shimfun');

addpath('/Users/stockmann/jason_dropbox/ASL_materials_lincoln_jason_yulin_meher/figures_code/origial_opt_code_from_nick');
addpath(genpath('/Users/stockmann/mri/mgh/matlab/yalmip/yalmip/YALMIP-master'));
addpath(genpath('/Users/stockmann/mri/mgh/matlab/mosek/9.2'))

scale_factor = 1;
max_current = 3*scale_factor;    
total_current = 40*scale_factor;
 tb=40;  excite_bw=405;   % to make solution easier, reduce tb and increase excite_bw
% tb=1;  excite_bw=350;

reject_frequency = -50;  % make sure this matches the reject_polarity
reject_polarity = 1;  % 1 to put reject lower than target, 0 to put reject higher than target, or -1?
  [shimmed,amps,success] = perform_shim_ncc_asl(shim.fmap,shim.mask,shim.coil.fm,max_current,total_current,tb,excite_bw,reject_frequency,reject_polarity);

  shim.territory_coil_shim_field_everywhere =  apply_current_using_coil_basis(amps, shim.coil.fm);

  
%    [shimmed,amps,success] = perform_shim_ncc_asl(shim.fmap,logical(shim.mask),shim.coil.fm,max_current,total_current,100,50,-175,1);
%     shimmed = perform_shim(shim.fmap,shim.mask,coil.fm,2,20);

if(~success)
    disp('------------------------------------')
    disp('constraints were NOT satisfiable')
    disp('------------------------------------')

else
    disp('------------------------------------')
    disp('constraints satisfied')
    disp('------------------------------------')
end

excite_b0 = shimmed(shim.mask==1);
reject_bw = shimmed(shim.mask==3);
all_voxels = [shimmed(shim.mask==1); shimmed(shim.mask==3)];
%
figure(80)
histogram(excite_b0); hold on;
histogram(reject_bw); hold off;

legend({'desired','reject'}); xlabel('Hz')
set(gca,'FontSize',28);




figure(81),histogram(all_voxels);
legend({'all arteries'}); xlabel('Hz')
set(gca,'FontSize',28);

%
% s = 40;
% figure(90)
% imagesc(double(mag_mask(:,:,s)).*shimmed(:,:,s),300*[-1,1]); colorcet('D1A');axis image off, %  h=colorbar,set(h,'FontSize',24,'Location','southoutside')
% add_mask_outline(logical(shim.mask(:,:,s)))
% 
% figure(91)
% imagesc(double(mag_mask(:,:,s)).*shim.fmap(:,:,s),300*[-1,1]); colorcet('D1A');axis image off,h=colorbar,set(h,'FontSize',24,'Location','southoutside')
% add_mask_outline(logical(shim.mask(:,:,s)))


% figure(92),
% subplot(1,2,1),overlay_color_on_grayscale(tof,shim.mask,shim.mask,[0 3] );
% subplot(1,2,2),overlay_color_on_grayscale(mag.*double(mag_mask),shim.mask,shim.mask,[0 3] );

figure(93),clf
imagesc3D((shim.territory_coil_shim_field_everywhere)),colormap('jet'),colorbar,set(gca,'CLim',[-400 400])

figure(94)
imagesc(vol2mos(shim.territory_coil_shim_field_everywhere),[-500 500]),colormap('jet'),colorbar, add_mask_outline(vol2mos(shim.mask));


disp('CURRENT TO USE FOR SHIM EXPERIMENT:')
out=sprintf('%4.2f, ' , amps)
disp(['Total current = ',num2str(sum(abs(amps)))])
shim.shimmed=shimmed;
shim.tof = tof;
%%
save ./outputs_for_figure/shim_all_vessels shim


%% plot localizer

% localizer = dicom_fm_import('LOCALIZER_0007');

% mag = dicom_fm_import('GRE_FIELD_75SLI_DSHIM_TRIG_0002');



% figure(100),imagesc(localizer(:,:,1)),axis image off,colormap(gray)
% figure(101),imagesc(rot90(squeeze(mag(:,round(end/2),:)))),colormap(gray),axis image off









































return


%% plot perfusion maps
inds=s-6:2:s+9;  plot_range = [0 350]
baseline_perfusion = dicom_fm_import('/Users/stockmann/mgh_data/shim_data/skyra_data/asl/subject_ASL_3T_3_GOOD_TERRITORY_MAPPING/TR5_3P75ISO_16MEAS_NOSHIM_MEANPERF_0016');
left_perfusion = dicom_fm_import('/Users/stockmann/mgh_data/shim_data/skyra_data/asl/subject_ASL_3T_3_GOOD_TERRITORY_MAPPING/TR5_3P75ISO_16MEAS_TERRITORY_MEANPERF_0019');
right_perfusion = dicom_fm_import('/Users/stockmann/mgh_data/shim_data/skyra_data/asl/subject_ASL_3T_3_GOOD_TERRITORY_MAPPING/TR5_3P75ISO_16MEAS_TERRI_LEFT_MEANPERF_0027');


figure(100),imagesc(baseline_perfusion(:,:,s),plot_range),colormap(gray),axis image off
figure(101),imagesc(left_perfusion(:,:,s),plot_range),colormap(gray),axis image off
figure(102),imagesc(right_perfusion(:,:,s),plot_range),colormap(gray),axis image off

figure(103),imagesc(vol2mos(baseline_perfusion(:,:,inds)),plot_range),colormap(gray),axis image off
figure(104),imagesc(vol2mos(left_perfusion(:,:,inds)),plot_range),colormap(gray),axis image off
figure(105),imagesc(vol2mos(right_perfusion(:,:,inds)),plot_range),colormap(gray),axis image off


%% --------------------------------------------------------------------




































% working version of old spectroscopy water-fat solver that does not hold the "TARGET" histogram at zero
max_current=2;
total_current=20;
[shimmed,amps,db0,mask_used,filt] = perform_shim_ncc_skull_linprog_ASL(shim.fmap,shim.mask,shim.coil.fm, max_current,total_current,opts);

%


disp(['TOTAL CURRENT FOR SEPARATION SHIM: ',num2str(sum(abs(amps)))])
shim.shimmed=shimmed;
% HOMOGENEITY SHIM
% shim.mask(shim.mask==3)=1;
%   [shimmed,amps,std_unshimmed,std_shimmed] = perform_shim(shim.fmap,shim.mask,shim.coil.fm,3,30);
% out=sprintf('%4.2f, ' , amps)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if compensate_B0 == 1
    total_current_B0.value = 20;  total_current_B0.mask = ones(1,numel(shim.coil.fm(1,1,1,:))).';
    max_current_B0 = 20*ones(1,numel(shim.coil.fm(1,1,1,:))).';
    simple_mask = shim.mask; simple_mask(simple_mask==3)=0;
    B0_offset = mean(mean(mean(shimmed(logical(simple_mask(:))))))
    disp(['Mean B0 offset: ',num2str(B0_offset)])
%     [shimmed_B0_offset,amps_B0_offset,std_unshimmed,std_shimmed] = perform_shim(B0_offset*ones(size(shimmed)),simple_mask,shim.coil.fm,20,200);
    [shimmed_B0_offset_temp,amps_B0_offset,std_unshimmed,std_shimmed] = perform_shim_arb_current_limits(B0_offset*ones(size(shimmed)),simple_mask,shim.coil.fm,max_current_B0,total_current_B0);
    shimmed_B0_offset = apply_current_using_coil_basis(amps_B0_offset, shim.coil.fm);
    amps = amps + amps_B0_offset;
    shim.shimmed=shim.shimmed + shimmed_B0_offset
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optional add B0 offset


shim.M=[-300:1:400];
shim=gather_histogram_values(shim);                    
plot_hist(shim)

figure(12),imagesc(vol2mos(shim.shimmed),[-500 500]),colormap(jet),axis image off

disp('CURRENT TO USE FOR SHIM EXPERIMENT:')
out=sprintf('%4.2f, ' , amps)
if sum(abs(amps)) > total_HARD_LIMIT_CURRENT, disp('ERROR: TOTAL CURRENT EXCEEDED'); end

save shim_tof shim











%% random figure


figure(1010),ch=[6 14 16 21];   tof_sli = 25;
% imagesc([squeeze(coil.fm(:,:,tof_sli,ch(1))).*double(tof_mask_slice) squeeze(coil.fm(:,:,tof_sli,ch(2))).*double(tof_mask_slice)]; [squeeze(coil.fm(:,:,tof_sli,ch(3))).*double(tof_mask_slice) squeeze(coil.fm(:,:,tof_sli,ch(4))).*double(tof_mask_slice)] ),

colorcet('D1A'),axis image off


%%
function shim = gather_histogram_values(shim)

temp=shim.fmap(shim.mask==1);   [shim.hist.unshimmed.want.N,shim.hist.unshimmed.want.X]=hist(temp,shim.M);
temp=shim.fmap(shim.mask==3);   [shim.hist.unshimmed.dont.N,shim.hist.unshimmed.dont.X]=hist(temp,shim.M);
                    
temp=shim.shimmed(shim.mask==1);   [shim.hist.shimmed.want.N,shim.hist.shimmed.want.X]=hist(temp,shim.M);
temp=shim.shimmed(shim.mask==3);   [shim.hist.shimmed.dont.N,shim.hist.shimmed.dont.X]=hist(temp,shim.M);
               
end


function plot_hist(shim)
figure(11),clf,subplot(1,2,1),

    plot(shim.hist.unshimmed.want.X,shim.hist.unshimmed.want.N,'LineWidth',2),hold on
    plot(shim.hist.unshimmed.dont.X,shim.hist.unshimmed.dont.N,'LineWidth',2),hold on,legend('want','dont want'),set(gca,'FontSize',20)
    
subplot(1,2,2),

    plot(shim.hist.shimmed.want.X,shim.hist.shimmed.want.N,'LineWidth',2),hold on
    plot(shim.hist.shimmed.dont.X,shim.hist.shimmed.dont.N,'LineWidth',2),hold on,legend('want','dont want'),set(gca,'FontSize',20)
    
end
                                        
