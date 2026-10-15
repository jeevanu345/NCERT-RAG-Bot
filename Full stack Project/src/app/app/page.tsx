import { redirect } from "next/navigation";
import { currentAccount } from "@/auth";
import { authenticationRequired } from "@/lib/auth-config";
import StudyWorkspace from "@/components/StudyWorkspace";
export const dynamic = "force-dynamic";
export const metadata = {
  title: "Study space",
  robots: { index: false, follow: false },
};
export default async function StudyPage() {
  const account = await currentAccount();
  if (!account && authenticationRequired())
    redirect("/sign-in?error=SessionRequired");
  return <StudyWorkspace key={account?.id || "guest"} account={account} />;
}
