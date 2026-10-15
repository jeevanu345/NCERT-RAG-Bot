import { useId } from "react";

export function LogoMark({ className = "" }: { className?: string }) {
  const id = useId().replaceAll(":", "");
  return (
    <svg
      className={`logo-mark ${className}`}
      viewBox="0 0 110 102"
      fill="none"
      aria-hidden="true"
    >
      <defs>
        <linearGradient
          id={`${id}-deep`}
          x1="4"
          y1="40"
          x2="91"
          y2="104"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#003e35" />
          <stop offset="1" stopColor="#002e28" />
        </linearGradient>
        <linearGradient
          id={`${id}-green`}
          x1="20"
          y1="30"
          x2="75"
          y2="94"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#00856b" />
          <stop offset="1" stopColor="#327b61" />
        </linearGradient>
        <linearGradient
          id={`${id}-sage`}
          x1="92"
          y1="42"
          x2="55"
          y2="99"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#639c7c" />
          <stop offset="1" stopColor="#3d8e71" />
        </linearGradient>
      </defs>
      <path
        d="M4 43.5C4 41.8 5.6 40.7 7.1 41.4L16 44.8V72C16 76 18.2 78.7 22 79.8C35.9 84 45.4 91.1 52.7 99.5C37.9 91.6 22.4 89.3 7.7 86C5.4 85.5 4 84.1 4 81.8V43.5Z"
        fill={`url(#${id}-deep)`}
      />
      <path
        d="M106 43.5C106 41.8 104.4 40.7 102.9 41.4L94 44.8V72C94 76 91.8 78.7 88 79.8C74.1 84 64.6 91.1 57.3 99.5C72.1 91.6 87.6 89.3 102.3 86C104.6 85.5 106 84.1 106 81.8V43.5Z"
        fill={`url(#${id}-deep)`}
      />
      <path
        d="M19.5 33.8C19.5 32.4 20.6 31.7 22.1 32.3C39.2 39.5 50.3 50.5 53 67.7C43.7 51.8 33.6 45.9 19.5 41.2V33.8Z"
        fill={`url(#${id}-green)`}
      />
      <path
        d="M90.5 33.8C90.5 32.4 89.4 31.7 87.9 32.3C70.8 39.5 59.7 50.5 57 67.7C66.3 51.8 76.4 45.9 90.5 41.2V33.8Z"
        fill={`url(#${id}-green)`}
      />
      <path
        d="M19.5 45.4C34.7 50.4 46.6 59.9 53 77.3V94.3C44.5 84.5 34.5 79 23.3 75.6C20.8 74.9 19.5 73.2 19.5 70.7V45.4Z"
        fill={`url(#${id}-green)`}
      />
      <path
        d="M90.5 45.4C75.3 50.4 63.4 59.9 57 77.3V94.3C65.5 84.5 75.5 79 86.7 75.6C89.2 74.9 90.5 73.2 90.5 70.7V45.4Z"
        fill={`url(#${id}-sage)`}
      />
      <circle cx="55" cy="37.5" r="8" fill="#007c65" />
      <path
        d="M55 10V20M36.5 19L42 25M73.5 19L68 25"
        stroke="#77a58c"
        strokeWidth="4.2"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function Logo({
  className = "",
  compact = false,
}: {
  className?: string;
  compact?: boolean;
}) {
  return (
    <span className={`brand-logo ${className}`}>
      <LogoMark />
      {!compact && (
        <span className="logo-wordmark">
          <strong>NCERT</strong>
          <span>tutor</span>
        </span>
      )}
    </span>
  );
}
