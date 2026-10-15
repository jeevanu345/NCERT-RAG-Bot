import { importPKCS8, SignJWT } from "jose";

export async function createAppleClientSecret({
  teamId,
  clientId,
  keyId,
  privateKey,
  days = 180,
  now = Math.floor(Date.now() / 1000),
}) {
  if (
    ![teamId, clientId, keyId].every(
      (value) => typeof value === "string" && /^[a-zA-Z0-9_.-]+$/.test(value),
    )
  )
    throw new Error("Provide a valid Apple Team ID, Services ID, and Key ID.");
  if (!Number.isInteger(days) || days < 1 || days > 180)
    throw new Error("Expiry must be between 1 and 180 days.");
  const expires = now + days * 86400;
  const key = await importPKCS8(privateKey, "ES256");
  const secret = await new SignJWT({})
    .setProtectedHeader({ alg: "ES256", kid: keyId })
    .setIssuedAt(now)
    .setExpirationTime(expires)
    .setIssuer(teamId)
    .setAudience("https://appleid.apple.com")
    .setSubject(clientId)
    .sign(key);
  return { secret, expires };
}
