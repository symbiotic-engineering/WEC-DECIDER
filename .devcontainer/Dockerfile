# Multi-stage build: Start with Gurobi, then layer on Calkit
FROM gurobi/optimizer:latest as gurobi-base
FROM ghcr.io/calkit/devcontainer:latest

# Copy Gurobi installation from official image
USER root
COPY --from=gurobi-base /opt/gurobi /opt/gurobi

# Set Gurobi environment variables (matching the official Gurobi image)
ENV GUROBI_HOME=/opt/gurobi/linux \
    PATH=$PATH:/opt/gurobi/linux/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/gurobi/linux/lib

# make Gurobi command line executable
RUN chmod +x /opt/gurobi/linux/bin/gurobi_cl

# Copy license setup script
COPY scripts/setup_gurobi_license.sh /usr/local/bin/setup_gurobi_license.sh
RUN chmod +x /usr/local/bin/setup_gurobi_license.sh

# Switch back to vscode user
USER vscode

# Set up license environment for the container startup
RUN echo 'source /usr/local/bin/setup_gurobi_license.sh' >> ~/.bashrc
