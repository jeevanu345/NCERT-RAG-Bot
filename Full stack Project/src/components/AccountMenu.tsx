"use client";
import Link from "next/link";
import { DropdownMenu } from "radix-ui";
import { ChevronDown, LogIn, LogOut, UserRound } from "lucide-react";
import type { AccountInfo } from "@/lib/auth-config";
export default function AccountMenu({
  account,
  compact = false,
}: {
  account: AccountInfo | null;
  compact?: boolean;
}) {
  if (!account)
    return (
      <Link
        className={`account-sign-in ${compact ? "compact" : ""}`}
        href="/sign-in"
        aria-label="Sign in"
      >
        <LogIn size={15} />
        <span>Sign in</span>
      </Link>
    );
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger
        className="account-trigger"
        aria-label="Open account menu"
      >
        <span className="account-initial">
          {account.name.slice(0, 1).toUpperCase()}
        </span>
        {!compact && <span>{account.name.split(" ")[0]}</span>}
        <ChevronDown size={13} />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="account-dropdown"
          align="end"
          sideOffset={12}
        >
          <div className="account-identity">
            <UserRound size={17} />
            <div>
              <strong>{account.name}</strong>
              {account.email && <p>{account.email}</p>}
            </div>
          </div>
          <DropdownMenu.Separator />
          <DropdownMenu.Item asChild>
            <Link href="/app">Study workspace</Link>
          </DropdownMenu.Item>
          <DropdownMenu.Item asChild>
            <Link href="/sign-out">
              <LogOut size={14} />
              Sign out
            </Link>
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
