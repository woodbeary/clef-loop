# The validator as a Cloudflare Container image. Node 22.18+ runs the TypeScript directly.
FROM node:22-slim
WORKDIR /app
COPY server.ts rules.ts ./
USER node
EXPOSE 8080
CMD ["node", "server.ts"]
