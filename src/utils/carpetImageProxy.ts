/**
 * Use proxy URL for remote carpet images so canvas can load them without CORS.
 * Same-origin and data URLs are returned unchanged.
 */
const PROXY_PATH = '/api/carpet-image'

export function getProxiedCarpetUrl(url: string): string {
  if (!url || url.startsWith('data:')) return url
  try {
    const parsed = new URL(url, window.location.origin)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return url
    if (parsed.origin === window.location.origin) return url
    return `${PROXY_PATH}?url=${encodeURIComponent(url)}`
  } catch {
    return url
  }
}
