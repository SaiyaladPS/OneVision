# Shared image storage: Car Scan and OneVision Report

Both apps use the same Central Storage Service as the durable source of image
bytes. Neither app needs to mount the other app's local image directory.

## Image lifecycle

1. Car Scan writes a new plate image to its local archive and records the
   relative path in the shared PostgreSQL `plates` row.
2. Its background storage worker uploads the image to
   `POST {STORAGE_SERVICE_BASE_URL}/api/files`, with the configured project and
   archive folder. The service returns a stable UUID.
3. Car Scan updates all matching image fields in PostgreSQL to
   `storage://<UUID>`. It only removes the local image after that DB update
   succeeds. If upload or DB linking fails, the local copy is retained and the
   worker retries.
4. OneVision Report reads that same PostgreSQL reference. Its server fetches
   `GET {STORAGE_SERVICE_BASE_URL}/api/files/<UUID>/download` using the private
   bearer token, then streams the image from its own `/api/plates/image` route.
   The browser never receives the Storage Service token.

The existing `GET /api/plates/image?id=<plate-id>&variant=crop` endpoint remains
the stable image URL for clients. `variant` accepts `full`, `crop`, or `ocr`.
Car Scan also serves saved images through its authenticated scan image routes.

## Configuration

Configure both apps to use the same reachable Storage Service, project, token,
and PostgreSQL database. The URL must be reachable from the app server/container,
not merely from a user's browser.

Car Scan `.env`:

```dotenv
DATABASE_URL=postgresql://...
STORAGE_SERVICE_BASE_URL=http://10.0.200.205:8080
STORAGE_SERVICE_PROJECT_CODE=onevision
STORAGE_SERVICE_API_TOKEN=<private project token>
```

OneVision Report `.env`:

```dotenv
DATABASE_URL=postgresql://...
STORAGE_SERVICE_BASE_URL=http://10.0.200.205:8080
STORAGE_SERVICE_PROJECT_CODE=onevision
STORAGE_SERVICE_API_TOKEN=<same private project token>
```

For local development, point both apps at the same reachable Storage Service
(for example, the LAN address above). For production, set these values in the
production secret store/container environment. Do not put the storage token in
`NUXT_PUBLIC_*`, frontend JavaScript, or a committed `.env` file. The image
proxy routes let clients in either environment use their app's HTTP URL.

## Existing archive migration

If Report can read the archive root containing `.storage_index.json`, it can
resolve older DB rows whose values are still local relative paths. New uploads
write portable `storage://` references directly into PostgreSQL. Keep the
archive mount read-only if desired; new image writes go to Central Storage.

The Report `/api/plates/image` upload route is retained for older Car Scan
deployments. It now uploads into Central Storage and updates the shared DB
reference rather than writing to the Report container's filesystem.
