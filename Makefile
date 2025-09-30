###############################################################################
# Directory and File Variables 
###############################################################################

TopDir          = ${CURDIR}
DockerDir       = $(TopDir)/docker
PyFiles         = $(wildcard $(adsp)/*.py)

# Run Python simulation scripts
docker-run:
	@bash $(DockerDir)/pysim-run.sh

# Launch Jupyter Notebook
nb:
	jupyter notebook --notebook-dir=./ --ip 0.0.0.0 --port 8888 --no-browser --allow-root

# Format Python files
py-format:
	@echo "Formatting Python files: $(PyFiles)"
	yapf -i $(PyFiles)

# Clean temporary and generated files
clean:
	@rm -rvf *.log ./build/ vivado* ./Xil *.str $(simCleans) *.txt urgReport $(localDbDir)/*/__pycache__ \
	vdCovLog *.conf $(pyDir)/__pycache__ $(pyDir)/.ipynb_checkpoints .Xil sim-* Q* *-test *.txt \
	*.svf

# Display help
###############################################################################
# Help Target
###############################################################################

.PHONY: help

help:
	@echo "######################################################################"
	@echo "#           Makefile   Help                                          #"
	@echo "######################################################################"
	@echo ""
	@echo "Available Targets:"
	@echo "  py-run                - Run Python simulations."
	@echo "  nb                    - Launch Jupyter Notebook."
	@echo "  clean                 - Clean temporary files."
	@echo "######################################################################"
