FROM python:3.9

RUN useradd -m -u 1000 aim_user

# Switch to the "aim_user" user
USER aim_user

# Set home to the user's home directory
ENV HOME=/home/aim_user \
	PATH=/home/aim_user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME

# install the `aim` package on the latest version
RUN pip install aim

ENTRYPOINT ["/bin/sh", "-c"]

COPY aim_repo.tar.gz .
RUN tar xvzf aim_repo.tar.gz
# have to run `aim init` in the directory that stores aim data for
# otherwise `aim up` will prompt for confirmation to create the directory itself.
# We run aim listening on 0.0.0.0 to expose all ports. Also, we run
# using `--dev` to print verbose logs. Port 43800 is the default port of
# `aim up` but explicit is better than implicit.
CMD ["aim up --host 0.0.0.0 --port 7860 --workers 2"]
