"use server";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { AuthError } from "next-auth";
import { signIn, signOut } from "@/auth";
import {
  authConfigured,
  providerAvailability,
  safeReturnTo,
} from "@/lib/auth-config";
export async function startSignIn(form: FormData) {
  const provider = form.get("provider");
  if (
    !authConfigured() ||
    !providerAvailability().some((p) => p.id === provider && p.enabled)
  )
    redirect("/sign-in?error=Configuration");
  try {
    await signIn(String(provider), {
      redirectTo: safeReturnTo(form.get("returnTo")),
    });
  } catch (error) {
    if (error instanceof AuthError)
      redirect(`/auth/error?error=${encodeURIComponent(error.type)}`);
    throw error;
  }
}
export async function finishSignOut() {
  // Rotate guest identity on sign-out so another person on this device cannot
  // inherit the previous guest workspace through an old cookie.
  (await cookies()).delete("ncert-workspace");
  if (authConfigured()) await signOut({ redirect: false });
  redirect("/sign-in?signedOut=1");
}
