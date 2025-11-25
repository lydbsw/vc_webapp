FROM node:18-bullseye

# Install Python and pip for the Python scorer
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy server package files and install Node deps
COPY server/package*.json ./
RUN npm ci --only=production

# Copy Python requirements (context should be repo root when building)
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the repository into the image
COPY . ./

# Server runs from /app/server
WORKDIR /app/server

EXPOSE 5001

# Start the Node server (it will spawn the Python scorer as needed)
CMD ["node", "index.js"]
