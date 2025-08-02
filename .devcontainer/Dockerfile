FROM ghcr.io/calkit/devcontainer:latest

# Copy Gurobi installation from official image
USER root
COPY --from=gurobi/optimizer:latest /opt/gurobi /opt/gurobi

# Set Gurobi environment variables (matching the official Gurobi image)
ENV GUROBI_HOME=/opt/gurobi/linux \
    PATH=$PATH:/opt/gurobi/linux/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/gurobi/linux/lib

# make Gurobi command line executable
RUN chmod +x /opt/gurobi/linux/bin/gurobi_cl

# Copy license setup script
COPY scripts/setup_gurobi_license.sh /usr/local/bin/setup_gurobi_license.sh
RUN chmod +x /usr/local/bin/setup_gurobi_license.sh

# Copy Julia and its env vars from image
COPY --from=julia:1.9.4-bullseye /usr/local/julia /usr/local/julia
ENV JULIA_PATH=/usr/local/julia \
    PATH=$PATH:/usr/local/julia/bin \
    JULIA_GPG=3673DF529D9049477F76B37566E3C7DC03D6E495 \
    JULIA_VERSION=1.9.4

# Switch back to vscode user
USER vscode

# Set up license environment for the container startup
RUN echo 'source /usr/local/bin/setup_gurobi_license.sh' >> ~/.bashrc
