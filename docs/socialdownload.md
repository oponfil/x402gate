# SocialDownload Provider

Download media from social networks via [RapidAPI Social Download All In One](https://rapidapi.com/nguyenmanhict-MuTUtGWD7K/api/social-download-all-in-one).

Supported platforms include YouTube, TikTok, Instagram, Twitter/X, VK, Rutube, Facebook, Dailymotion, Twitch, Vimeo, Reddit, Pinterest, Snapchat, Bilibili, LinkedIn, and others — any URL that RapidAPI's service supports.

## Endpoint

```
POST /v1/socialdownload/download
```

## Pricing

Fixed **$0.005 per request** (all platforms).

## Request Format

```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | ✅ | URL of the social media post/video to download |

## Response Format

The gateway returns metadata and a list of available media variants. The `best_media` field highlights the recommended download (highest-resolution combined MP4 when available).

```json
{
  "data": {
    "title": "Video Title",
    "author": "Channel Name",
    "source": "youtube",
    "duration": 210,
    "thumbnail": "https://i.ytimg.com/vi/.../hqdefault.jpg",
    "best_media": {
      "url": "https://direct-download-url.mp4",
      "type": "video",
      "extension": "mp4",
      "quality": "720p",
      "width": 1280,
      "height": 720
    },
    "medias": [
      {
        "url": "https://...",
        "type": "video",
        "extension": "mp4",
        "quality": "1080p",
        "width": 1920,
        "height": 1080,
        "is_audio": false
      },
      {
        "url": "https://...",
        "type": "video",
        "extension": "mp4",
        "quality": "720p",
        "width": 1280,
        "height": 720,
        "is_audio": true
      },
      {
        "url": "https://...",
        "type": "audio",
        "extension": "mp3",
        "quality": "128kbps"
      }
    ]
  }
}
```

### Response Fields

| Field | Type | Description |
|---|---|---|
| `title` | string | Media title |
| `author` | string | Author/channel name |
| `source` | string | Platform identifier (youtube, tiktok, instagram, etc.) |
| `duration` | int\|null | Duration in seconds (if available) |
| `thumbnail` | string | Thumbnail URL |
| `best_media` | object | Recommended download (highest-res combined MP4) |
| `medias` | array | All available media variants |

### Media Object Fields

| Field | Type | Description |
|---|---|---|
| `url` | string | Direct download URL |
| `type` | string | `video`, `audio`, or `image` |
| `extension` | string | File extension (`mp4`, `mp3`, `jpg`, etc.) |
| `quality` | string | Quality label (e.g. `720p`, `1080p`) |
| `width` | int | Video width in pixels |
| `height` | int | Video height in pixels |
| `is_audio` | bool | `true` = combined video+audio track |

## Media Selection Logic

The `best_media` field is selected automatically:

1. **Combined MP4** (video+audio, `is_audio: true`) with the highest resolution
2. **Any MP4 video** with the highest resolution (if no combined available)
3. **Any media** with a download URL (fallback)

Clients can also inspect the full `medias` array and pick a different variant.

## Example

```bash
curl -X POST https://x402gate.io/v1/socialdownload/download \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

## Notes

- This is a **synchronous** provider — the response is returned immediately (no polling).
- The gateway returns **direct download URLs** — the client downloads the media file itself.
- Download URLs are temporary and may expire. Download promptly after receiving the response.
- The provider does not validate URL domains — any URL is forwarded to RapidAPI, which decides whether it's supported.
