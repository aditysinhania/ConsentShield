import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export function HomePage() {
  return (
    <section className="relative overflow-hidden rounded-3xl border border-moss/10 bg-meadow px-8 py-16 md:px-14 md:py-20">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-2xl"
      >
        <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-moss">
          Consent integrity
        </p>
        <h1 className="font-display text-5xl font-bold leading-[1.05] tracking-tight text-ink md:text-6xl">
          ConsentShield
        </h1>
        <p className="mt-5 max-w-xl text-lg text-ink/75">
          Detect verifiable consent dark patterns in cookie banners and subscription flows —
          with evidence you can audit, not black-box guesses.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            to="/scan"
            className="rounded-xl bg-moss px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-moss/90"
          >
            Open scanner
          </Link>
          <Link
            to="/history"
            className="rounded-xl border border-ink/15 bg-white/50 px-5 py-3 text-sm font-semibold text-ink transition hover:bg-white"
          >
            View history
          </Link>
        </div>
      </motion.div>
      <motion.div
        aria-hidden
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.15, duration: 0.6 }}
        className="pointer-events-none absolute -right-10 bottom-0 hidden h-64 w-64 rounded-full bg-clay/20 blur-2xl md:block"
      />
    </section>
  );
}
