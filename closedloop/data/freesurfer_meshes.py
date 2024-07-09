import os
import os.path as op
import sys
import subprocess
import multiprocessing


def compute_freesurfer(freesurfer_home, subjects_dir, 
                       subject_name, mri_fname, n_jobs):
    """This function allows to launch the complete freesurfer segmentation 
        pipeline on a single subject, it will generate a shell file containing
        the information, execute it, and finally delete it.

    Args:
        freesurfer_home (path_like): Path to the freesurfer home
        subjects_dir (path_like): Path to the subjects foleder
        subject_name (str): Name of the subject to process
        mri_fname (path_like): Path to the .nii MRI file
        n_jobs (int): Number of threads to use, can be 1 to max(threads), 
            or -1 to consider all threads
    """
    # Bash does not account -1 as the max num of cores, doing it manually
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    # Define bash file name
    cmd_file = op.join(os.getcwd(), 'fs_launcher_{0}.sh'.format(subject_name))
    # Write commands
    with open(cmd_file, 'w') as f:
        f.writelines("#!/bin/bash\n")
        f.writelines('export FREESURFER_HOME={0}\n'.format(freesurfer_home))
        f.writelines('source $FREESURFER_HOME/SetUpFreeSurfer.sh\n')
        f.writelines('export SUBJECTS_DIR={0}\n'.format(subjects_dir))
        f.writelines('recon-all -subjid {0} -i {1} -all -parallel -openmp {2}'
                     .format(subject_name, mri_fname, n_jobs))
    # Change bash file rights for execution
    os.system('chmod +x {0}'.format(cmd_file))
    # Run the bash script
    # subprocess.run('source {0}'.format(cmd_file), shell=True, text=True, 
    #                executable='/bin/bash')
    subprocess.check_call('source {0}'.format(cmd_file), shell=True, text=True,
                          executable='/bin/bash')
    # Remove bash file
    os.remove(cmd_file)

    return


def compute_VEP(vep_dir, freesurfer_home, subjects_dir, subject_name):
    # Add VEP scripts to environmental paths
    sys.path.insert(1, vep_dir)
    import convert_to_vep_parc
    # Create cortical VEP parcellation for left and right hemispheres
    for h in ['lh', 'rh']:
        convert_to_vep_parc.convert_parc(
            '{0}/{1}/label/{2}.aparc.a2009s.annot'.format(subjects_dir, 
                                                          subject_name, h),
            '{0}/{1}/surf/{2}.pial'.format(subjects_dir, subject_name, h),
            '{0}/{1}/surf/{2}.inflated'.format(subjects_dir, subject_name, h),
            '{0}'.format(h),
            '{0}/data/VepAparcColorLut.txt'.format(vep_dir),
            '{0}/data/VepAtlasRules.txt'.format(vep_dir),
            '{0}/{1}/label/{2}.aparc.vep.annot'.format(subjects_dir, 
                                                       subject_name, h)
        )
    # Define bash file name
    cmd_file = op.join(os.getcwd(), 'vep_parc_{0}.sh'.format(subject_name))
    # Write commands
    with open(cmd_file, 'w') as f:
        f.writelines("#!/bin/bash\n")
        f.writelines('export SUBJECTS_DIR={0}\n'.format(subjects_dir))
        f.writelines('export FREESURFER_HOME={0}\n'.format(freesurfer_home))
        f.writelines('source $FREESURFER_HOME/SetUpFreeSurfer.sh\n')
        f.writelines('mri_aparc2aseg --s {0} '.format(subject_name) + 
                     '--annot aparc.vep --base-offset 70000 ' + 
                     '--o {0}/{1}/mri/aparc+aseg.vep.mgz'.format(subjects_dir,
                                                                 subject_name))
    # Change bash file rights for execution
    os.system('chmod +x {0}'.format(cmd_file))
    # Run the bash script
    subprocess.check_call('source {0}'.format(cmd_file), shell=True, text=True,
                          executable='/bin/bash')
    # Remove bash file
    os.remove(cmd_file)
    # Create volumetric subcortical VEP parcellation
    convert_to_vep_parc.convert_seg(
        '{0}/{1}/mri/aparc+aseg.vep.mgz'.format(subjects_dir, subject_name),
        '{0}/data/VepFreeSurferColorLut.txt'.format(vep_dir),
        '{0}/data/VepAtlasRules.txt'.format(vep_dir),
        '{0}/{1}/mri/aparc+aseg.vep.mgz'.format(subjects_dir, subject_name)
    )
    
    return


def compute_watershed_bem(freesurfer_home, subjects_dir, subject_name):
    
    # Define bash file name
    cmd_file = op.join(os.getcwd(), 'wtshd_bem_{0}.sh'.format(subject_name))
    # Write commands
    with open(cmd_file, 'w') as f:
        f.writelines('#!/bin/bash\n')
        f.writelines('source activate py310\n')
        f.writelines('export SUBJECTS_DIR={0}\n'.format(subjects_dir))
        f.writelines('export FREESURFER_HOME={0}\n'.format(freesurfer_home))
        f.writelines('source $FREESURFER_HOME/SetUpFreeSurfer.sh\n')
        f.writelines('mne watershed_bem -o -a -p 15 -t true ' +
                     '-s {0} '.format(subject_name) +
                     '-d {0} '.format(subjects_dir) +
                     '--verbose \n')
        # Optional, perform surface smoothing
        f.writelines('mris_smooth -a 10 -n 10 ' +
                     '{0}/{1}/bem/watershed/{1}_inner_skull_surface '.format(
                         subjects_dir, subject_name) +
                     '{0}/{1}/bem/watershed/{1}_inner_skull_surface'.format(
                         subjects_dir, subject_name))
    # Change bash file rights for execution
    os.system('chmod +x {0}'.format(cmd_file))
    # Run the bash script
    subprocess.check_call('source {0}'.format(cmd_file), shell=True, text=True,
                          executable='/bin/bash')
    # Remove bash file
    os.remove(cmd_file)

    return


def compute_scalp_meshes(freesurfer_home, subjects_dir, subject_name):
    """This function allows to launch the complete mne pipeline for scalp 
        meshes reconstruction on a single subject, it will generate a shell 
        file containing the information, execute it, and finally delete it.

    Args:
        freesurfer_home (path_like): Path to the freesurfer home
        subjects_dir (path_like): Path to the subjects foleder
        subject_name (str): Name of the subject to process
    """
    
    # Define bash file name
    cmd_file = op.join(os.getcwd(), 'scalp_meshes_{0}.sh'.format(subject_name))
    # Write commands
    with open(cmd_file, 'w') as f:
        f.writelines('#!/bin/bash\n')
        f.writelines('source activate py310\n')
        f.writelines('export SUBJECTS_DIR={0}\n'.format(subjects_dir))
        f.writelines('export FREESURFER_HOME={0}\n'.format(freesurfer_home))
        f.writelines('source $FREESURFER_HOME/SetUpFreeSurfer.sh\n')
        f.writelines('mne make_scalp_surfaces -on ' +
                     '-s {0} '.format(subject_name) +
                     '-d {0} '.format(subjects_dir) +
                     '--verbose')
    # Change bash file rights for execution
    os.system('chmod +x {0}'.format(cmd_file))
    # Run the bash script
    subprocess.check_call('source {0}'.format(cmd_file), shell=True, text=True,
                          executable='/bin/bash')
    # Remove bash file
    os.remove(cmd_file)

    return


if __name__ == '__main__':
    
    prj_data = '/home/ruggero.basanisi/data/tweakdreams'
    fs_home = '/home/programmi/freesurfer'
    
    # subjects = ['TD001', 'TD002', 'TD003', 'TD005', 'TD006', 'TD007', 'TD008',
    #             'TD009', 'TD010', 'TD011', 'TD012', 'TD015', 'TD018', 'TD019', 
    #             'TD021', 'TD022', 'TD025', 'TD026', 'TD027', 'TD028', 'TD030', 
    #             'TD031', 'TD0032', 'TD034',]
    
    subjects = ['TD010', 'TD011']

    sbj_dir = op.join(prj_data, 'freesurfer')
    n_jobs = 32
    
    for sbj in subjects:
        mri_fname = op.join(prj_data, 'mri', 
                            f'sub-{sbj.lower()}_ses-d01_mri.nii')

        compute_freesurfer(fs_home, sbj_dir, sbj, mri_fname, n_jobs)
        compute_watershed_bem(fs_home, sbj_dir, sbj)
        compute_scalp_meshes(fs_home, sbj_dir, sbj)
    
    # 'mri_watershed -useSRAS -atlas -surf $SUBJECTS_DIR/$SUBJECT/bem/watershed $SUBJECTS_DIR/$SUBJECT/mri/T1.mgz $SUBJECTS_DIR/$SUBJECT/bem/TD005-head.fif'
    # 'mri_watershed -useSRAS -atlas -surf /home/jerry/freesurfer/TweakDreams/TD005/bem/watershed /home/jerry/freesurfer/TweakDreams/TD005/mri/T1.mgz /home/jerry/freesurfer/TweakDreams/TD005/bem/TD001-head.fif'

