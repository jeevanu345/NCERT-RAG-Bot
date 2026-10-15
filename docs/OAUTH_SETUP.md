# Google, Microsoft, and Apple sign-in

The app uses Auth.js for the OAuth/OIDC authorization-code flow, provider token validation, CSRF protection, encrypted HTTP-only session cookies, and sign-out. No provider tokens are sent to browser components or stored in the textbook database. The implementation is in `Full stack Project/src/auth.ts`; browser routes are under `/api/auth/*`.

Provider buttons become available only after their server credentials and a sufficiently long `AUTH_SECRET` are configured. Apple also requires an HTTPS `AUTH_URL`. Until then, the UI explains which options are unavailable. Guest study remains available by default.

## Common setup

1. Install the app with `npm run setup` at the repository root.
2. If you do not already have `Full stack Project/.env.local`, copy `Full stack Project/.env.example` to that path. Do not overwrite an existing configuration.
3. Run `npm run auth:secret`. This generates a random session secret in `.env.local` without printing it. It leaves an existing secret unchanged. Keep that secret stable across restarts and app instances; rotation signs everyone out.
4. Set `AUTH_URL` to the exact browser-facing origin, such as `http://localhost:3000` for Google/Microsoft local development or `https://study.example.com` for a hosted app. Open that same address in your browser. Register matching callback URLs below; hostnames, ports, paths, and schemes must match.
5. Fill in each provider's credentials on the server. Never use `NEXT_PUBLIC_` for credentials and never paste them into chat or commit them. Restart the app after configuration changes.

| Provider  | Server variables                                                                                          | Registered redirect/callback URL                                 |
| --------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Google    | `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`                                                                    | `https://study.example.com/api/auth/callback/google`             |
| Microsoft | `AUTH_MICROSOFT_ENTRA_ID_ID`, `AUTH_MICROSOFT_ENTRA_ID_SECRET`, optional `AUTH_MICROSOFT_ENTRA_ID_ISSUER` | `https://study.example.com/api/auth/callback/microsoft-entra-id` |
| Apple     | `AUTH_APPLE_ID`, `AUTH_APPLE_SECRET`                                                                      | `https://study.example.com/api/auth/callback/apple`              |

For Google/Microsoft locally, replace the example origin with `http://localhost:3000`. Apple web sign-in requires a registered HTTPS domain and does not support localhost callbacks. See the [Auth.js Apple provider requirements](https://authjs.dev/getting-started/providers/apple).

## Google

In [Google Cloud Console](https://console.cloud.google.com/apis/credentials), configure an OAuth consent screen and create an OAuth client of type **Web application**. Register the callback above, add test users while the consent screen is in testing, and copy its client ID and secret into the Google variables. The app requests only `openid email profile`.

Use Google's consent/verification process before opening the app beyond your allowed test users. See [the Auth.js Google setup](https://authjs.dev/getting-started/providers/google).

## Microsoft

In [Microsoft Entra admin center](https://entra.microsoft.com), create an **App registration**. Choose the supported account types appropriate to your audience. Add a **Web** platform with the callback above. Copy **Application (client) ID** into `AUTH_MICROSOFT_ENTRA_ID_ID`. Create a client secret and copy its **value**, not its secret ID, into `AUTH_MICROSOFT_ENTRA_ID_SECRET`.

The default issuer is `https://login.microsoftonline.com/common/v2.0`. Set `AUTH_MICROSOFT_ENTRA_ID_ISSUER=https://login.microsoftonline.com/YOUR_TENANT_ID/v2.0` for a tenant-specific app. The issuer must agree with the account types in the app registration. Track your client-secret expiration date and renew it before expiry. See [the Auth.js Microsoft Entra ID setup](https://authjs.dev/getting-started/providers/microsoft-entra-id).

## Apple

In [Apple Developer identifiers](https://developer.apple.com/account/resources/identifiers/list), enable Sign in with Apple for an App ID. Create a **Services ID**, configure it for web authentication with that primary App ID, and register the HTTPS domain and return URL. The Services ID is `AUTH_APPLE_ID`; it is not the native app bundle ID unless your registration explicitly uses that identifier as the web client.

Create a Sign in with Apple key and download its `.p8` file once. Keep it outside the repository. Note the Key ID and Team ID. Generate an ES256 client-secret JWT with the included helper:

```bash
# Run at the repository root; use your real identifiers and private key path.
npm run auth:apple-secret -- \
  --team-id YOUR_TEAM_ID \
  --client-id com.example.study.web \
  --key-id YOUR_KEY_ID \
  --private-key /private/path/AuthKey.p8
```

The helper writes `Full stack Project/.secrets/apple-client-secret.env` with private permissions and prints only its path and expiry. Copy the `AUTH_APPLE_SECRET` entry from that private file into `.env.local` or your hosting platform's secret manager. The `.p8` contents never go in `AUTH_APPLE_SECRET`. Generated secrets expire after 180 days by default. Renew before expiry; use `--days 30` for a shorter lifetime and `--out .secrets/apple-renewed.env` to choose a new output file. Existing output files are never overwritten.

Apple uses a cross-site form POST for its callback. On HTTPS, the app configures short-lived, secure OAuth state/nonce/PKCE cookies with `SameSite=None` so this flow can finish; the encrypted session cookie retains Auth.js defaults. Apple's first sign-in may provide a name that later callbacks omit. Hide My Email is supported through the standard Apple identity response. See [Auth.js Apple](https://authjs.dev/getting-started/providers/apple) and [Apple token validation/client-secret documentation](https://developer.apple.com/documentation/signinwithapplerestapi/generate-and-validate-tokens).

## Account and data behavior

- Sign-in links a server-side textbook library to the verified **provider and subject**. It does not use an email or a browser-supplied workspace header as authentication. Google, Microsoft, and Apple identities remain separate even if their emails match; no automatic account linking is performed.
- Account libraries and guest libraries occupy separate namespaces. Existing guest textbook libraries remain intact. Signing in does not silently merge guest books or conversations; upload books in the account workspace when needed.
- Chat history stays in local storage on that browser, scoped to the account. It does not sync to other devices. Themes/fonts are device preferences shared across account and guest screens. NVIDIA keys remain separate from OAuth credentials and are held in memory unless tab storage is explicitly selected.
- Session cookies last up to seven days. Sign-out clears the local session cookie and remembered tab key, rotates the guest workspace cookie, and offers to remove the account's cached chats. Other open tabs receive a sign-out signal. Signing out of this app does not sign out of Google/Microsoft/Apple globally.
- Without a session database, sign-out cannot revoke a copied session cookie on another device. Use a database-backed session strategy if central revocation and account administration are required.

## Hosting configuration

Set `AUTH_REQUIRED=true` to require sign-in for `/app` and the study API. This fails closed if no provider is ready: students see the sign-in page and study API requests receive 401. The landing page, privacy page, health endpoint, and auth routes stay available.

Use HTTPS, set the canonical `AUTH_URL`, and preserve the browser-facing host/forwarded host at your trusted reverse proxy. The app trusts that proxy for Auth.js host handling. Keep the Python API private; it trusts only the Next.js proxy's workspace identity/kind headers and is not an internet-facing authentication boundary. Use persistent private storage for SQLite, backups, upload/request limits, and hosting-level abuse controls. A server-wide `NVIDIA_API_KEY` permits every allowed user to spend its allowance; browser BYOK is the default approach.

## Verification

```bash
npm test
npm run lint
npm run typecheck
npm run build
npm run test:e2e
```

Browser tests start an isolated production server on ports 3100/8100 with synthetic credentials. They verify provider availability, CSRF rejection, real Auth.js encrypted sessions, account/guest isolation, sign-out cleanup, invalid-session rejection, and an ES256 Apple-secret signature. They do not contact providers or pretend to complete a live OAuth consent flow. To activate and verify live OAuth, configure your own registered clients and complete a sign-in/sign-out round trip with each provider on its registered origin.

For failures, check the host, callback path, supported Microsoft account types, Google consent-screen test users, Apple Services ID/domain/key/expiry, and system clock. The custom error page keeps secret-bearing provider errors out of the UI. Do not share server logs without redacting credentials.
