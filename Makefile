# Makefile
# Daniel Cairns, Jakob Thoms, Shirley Zhang
# 2023/06/27

# SLIPP - MDS Capstone

# This Makefile contains commands to run the preprocessing script and to generate our final report. 
# Other command line functions (such as those that train models) should be run 
# explicitly in the command line with desired flags (see README for examples).

# Make clean deletes the preprocessed data files and the final report PDF.

# Example usage:
# make clean
# make report
# make preprocess

# Make keywords

clean:
	rm -rf docs/final_report/final_report.pdf
	rm -rf data/preprocessed/*.nc
	
report: docs/final_report/final_report.pdf

preprocess: data/preprocessed/preprocessed_train_ds.nc

# Create the report:
docs/final_report/final_report.pdf: docs/final_report/final_report.Rmd
	Rscript -e "rmarkdown::render('docs/final_report/final_report.Rmd')"

# Pre-process the data:
data/preprocessed/preprocessed_train_ds.nc: data/IsoGSM/Total.IsoGSM.ERA5.monmean.nc data/IsoGSM/IsoGSM_land_sea_mask.nc data/IsoGSM/IsoGSM_orogrd.nc
	python src/Preprocessing/main.py -o 'data/preprocessed/'
