FROM python:3.9.12
		
ENV APP_HOME /finalproj
WORKDIR $APP_HOME
ENV PYTHONPATH /
	
# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*
	
# Get necessary python libraries
COPY requirements.txt .
# install mariaDB here, otherwise get version incompatible error
RUN pip3 install mariadb==1.0.11
RUN pip install --upgrade pip
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
COPY baseball.sql .
COPY app ./app
COPY sql ./sql
COPY startbash.sh .

RUN chmod +x ./startbash.sh
CMD ["./startbash.sh"]
