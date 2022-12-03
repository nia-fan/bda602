FROM python:3.6.9
		
ENV APP_HOME /hw6	
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
	
COPY baseball.sql .
COPY hw6.sql .
COPY startbash.sh .

CMD ["./startbash.sh"]