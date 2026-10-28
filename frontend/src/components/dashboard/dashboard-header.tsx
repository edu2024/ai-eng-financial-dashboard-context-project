import { Activity, LayoutDashboard } from 'lucide-react'

export type DashboardPage = 'dashboard' | 'telemetry'

interface DashboardHeaderProps {
  page: DashboardPage
  onPageChange: (page: DashboardPage) => void
  period?: string
}

export function DashboardHeader({
  page,
  onPageChange,
  period = '2024 — Full Year',
}: DashboardHeaderProps) {
  const isTelemetry = page === 'telemetry'

  return (
    <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
      <div className="flex items-center gap-3">
        <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
          {isTelemetry ? <Activity size={18} /> : <LayoutDashboard size={18} />}
        </span>
        <div>
          <h1 className="text-xl font-semibold text-foreground tracking-tight">
            {isTelemetry ? 'Telemetry Report' : 'Financial Overview'}
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            {isTelemetry ? 'Operational metrics dashboard' : 'Executive metrics dashboard'}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        {!isTelemetry ? (
          <span className="inline-flex items-center rounded-full border border-border bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground">
            {period}
          </span>
        ) : null}
        <nav aria-label="Dashboard navigation" className="flex items-center gap-1 rounded-lg border border-border bg-secondary/60 p-1">
          <button
            type="button"
            aria-pressed={page === 'dashboard'}
            onClick={() => onPageChange('dashboard')}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              page === 'dashboard'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Financial
          </button>
          <button
            type="button"
            aria-pressed={isTelemetry}
            onClick={() => onPageChange('telemetry')}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              isTelemetry
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Telemetry
          </button>
        </nav>
      </div>
    </header>
  )
}
