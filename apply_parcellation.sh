# apply_parcellation.sh
# loop through all subdirectories (subject directories)
# create a directory called 'hyperalignment_input'
# apply the glasser parcelation to all filtered dtseries
# and save as a nii using the workbench commands
# later we'll load in numpy and resave

BASEDIR="/gpfs/milgram/project/casey/ABCD_hyperalignment/data/derivatives/abcd-hcp-pipeline"
SUBPATH="ses-baselineYear1Arm1/func/"
OUTDIR="/gpfs/milgram/project/casey/ABCD_hyperalignment/data/hyperalignment_input"
PARCELLATION="/gpfs/milgram/project/casey/ABCD_hyperalignment/parcellations/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"

for dir in ${BASEDIR}/*/; 
do 
	SUBID=`basename "$dir"`
	DATADIR="${BASEDIR}/${SUBID}/${SUBPATH}/"
	INFILE=$(ls ${DATADIR}*task-rest_bold_desc-filtered_timeseries.dtseries.nii)
	OUTDN="${OUTDIR}/${SUBID}/"
	mkdir ${OUTDN}
	OUTFN="${OUTDN}/${SUBID}_rest_bold_filtered_timeseries_glasser_parcellated.ptseries.nii"
	wb_command -cifti-parcellate ${INFILE} ${PARCELLATION} 2 ${OUTFN}
	echo "${OUTFN}"
done
