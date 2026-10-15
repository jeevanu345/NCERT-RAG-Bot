import { currentAccount } from "@/auth";
import LandingExperience from "@/components/LandingExperience";
import { authenticationRequired } from "@/lib/auth-config";
export default async function LandingPage() {
  return (
    <LandingExperience
      account={await currentAccount()}
      guestAllowed={!authenticationRequired()}
    />
  );
}
