import react from '@vitejs/plugin-react'
import tailwind from 'tailwindcss'
import { defineConfig } from 'vite'

/** Proxy remote carpet images to avoid CORS when drawing on canvas (dev only). Production: host same endpoint elsewhere. */
function carpetImageProxy() {
  return {
    name: 'carpet-image-proxy',
    configureServer(server: { middlewares: { use: (fn: (req: any, res: any, next: () => void) => void) => void } }) {
      server.middlewares.use(async (req: any, res: any, next: () => void) => {
        if (!req.url?.startsWith('/api/carpet-image')) {
          next()
          return
        }
        try {
          const url = new URL(req.url, 'http://localhost').searchParams.get('url')
          if (!url) {
            res.statusCode = 400
            res.end('Missing url')
            return
          }
          const response = await fetch(url)
          const buffer = await response.arrayBuffer()
          const contentType = response.headers.get('content-type') ?? 'image/jpeg'
          res.setHeader('Content-Type', contentType)
          res.setHeader('Access-Control-Allow-Origin', '*')
          res.end(Buffer.from(buffer))
        } catch (err) {
          res.statusCode = 502
          res.end('Proxy failed')
        }
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), carpetImageProxy()],
  base: '/',
  css: {
    postcss: {
      plugins: [tailwind()],
    },
  },
  assetsInclude: ['**/*.wasm'],
})
