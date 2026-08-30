# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package file and npm config
COPY package.json ./
COPY .npmrc ./

RUN npm install --ignore-scripts

COPY . .

RUN npm run build

# Production stage - minimal Alpine image
FROM node:20-alpine

WORKDIR /app

# Copy package file and npm config
COPY package.json ./
COPY .npmrc ./

RUN npm install --ignore-scripts --omit=dev

# Copy built application
COPY --from=builder /app/dist ./dist

USER node

CMD ["npx", "vite", "preview", "--host", "--port", "80"]