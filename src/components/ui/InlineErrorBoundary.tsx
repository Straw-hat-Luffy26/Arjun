import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

/**
 * A boundary the size of one block, for the blocks inside a message.
 *
 * ## Why the app-level boundary was not enough
 *
 * `AppShell` wraps `<Outlet />` in an `ErrorBoundary`, which is the right thing
 * for a page that cannot render. It is the wrong granularity for a list of
 * produced files: when one artifact row threw — a briefing deck, whose kind the
 * icon table had no entry for — the boundary that caught it was the one around
 * the entire page, so a single unrecognised row replaced the whole conversation
 * with a crash card.
 *
 * The build spec asks for this directly: one bad artifact must not blank the
 * message stream. So the failure is caught next to the thing that failed, and
 * costs one row.
 *
 * ## What it is not
 *
 * Not a way to make render errors quiet. `componentDidCatch` logs with the
 * label it was given, so a row that fails still says so in the console and on
 * screen; it simply stops taking its neighbours with it. A boundary that
 * rendered `null` would hide the same defect this one exists to expose.
 */
interface Props {
  children: ReactNode;
  /** What failed, in the words the console and the reader should see. */
  label: string;
}

interface State {
  error: Error | null;
}

export class InlineErrorBoundary extends Component<Props, State> {
  public state: State = { error: null };

  public static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  public componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`Could not render ${this.props.label}:`, error, info);
  }

  public render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div
        role="alert"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 8px',
          fontSize: '12px',
          color: 'var(--text-secondary)',
        }}
      >
        <AlertTriangle size={12} color="var(--warning)" />
        <span>
          {this.props.label} could not be displayed
          {error.message ? ` (${error.message})` : ''}.
        </span>
      </div>
    );
  }
}
