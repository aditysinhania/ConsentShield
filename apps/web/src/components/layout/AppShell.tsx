import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";

const links = [
  { to: "/", label: "Home" },
  { to: "/scan", label: "Scan" },
  { to: "/history", label: "History" },
  { to: "/analytics", label: "Analytics" },
  { to: "/settings", label: "Settings" },
];

export function AppShell() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-moss/15 bg-sand/70 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-6 py-4">
          <NavLink to="/" className="font-display text-2xl font-bold tracking-tight text-ink">
            ConsentShield
          </NavLink>
          <nav className="flex flex-wrap gap-1">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  clsx(
                    "rounded-lg px-3 py-1.5 text-sm font-semibold transition",
                    isActive ? "bg-moss text-white" : "text-ink/70 hover:bg-mist",
                  )
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
