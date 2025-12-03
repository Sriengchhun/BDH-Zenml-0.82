FROM sriengchhun/bdh-zenml:0.80

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt ./ 
COPY . /app

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip \
#  && pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Update and install any system dependencies (if needed)

RUN rm -rf /var/lib/apt/lists/*  

EXPOSE 9002