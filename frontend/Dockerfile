# Development Dockerfile for Testing & Hot-Reload
FROM node:20-alpine

WORKDIR /app

# Install deps
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Expose Vite port
EXPOSE 5173

# Start Dev Server (Host 0.0.0.0 is needed for Docker)
CMD ["npm", "run", "dev", "--", "--host"]