# Frontend Dockerfile - Multi-stage build
# Build stage
FROM node:20-alpine AS builder

# Set work directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (use npm ci for reproducible builds)
RUN npm ci

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine AS production

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user for nginx
RUN addgroup -g 1001 -S nginx-user && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G nginx-user -g nginx-user nginx-user

# Fix permissions
RUN chown -R nginx-user:nginx-user /usr/share/nginx/html /var/cache/nginx /var/log/nginx /etc/nginx/conf.d /tmp

# Store the master PID in a location writable by the non-root user.
RUN sed -i 's#^pid .*;#pid /tmp/nginx.pid;#' /etc/nginx/nginx.conf

# Switch to non-root user
USER nginx-user

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:8080/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]