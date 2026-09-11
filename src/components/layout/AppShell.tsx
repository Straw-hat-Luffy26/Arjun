import React from 'react';
import { Outlet } from 'react-router-dom';
import { AppMenu } from './AppMenu';
import { ChatHeader } from '../chat/ChatHeader';
import { ErrorBoundary } from '../ui';
import { useLocation } from 'react-router-dom';
import { useConversation } from '../run/useConversation';
import { useAdoptedRun, useContextLedger } from '../run/runAdopt';
import { useMemo } from 'react';
import styles from './AppShell.module.css';

/**
 * The shell. Three vertical regions:
 *
 *   1. Top bar — the ARJUN wordmark (left), the conversation breadcrumb
 *      (center) and a quiet status pill (right). The wordmark is the
 *      history/advanced-navigation entry point. There is no bottom
 *      status bar; status lives in the top bar.
 *   2. Main — the routed page (Workbench, Tasks, etc). Workbench renders
 *      its own ChatHeader inside the chat column; the breadcrumb above
 *      doubles as a header for non-chat pages.
 *   3. (nothing) — the previous status bar is removed. The top bar is
 *      enough.
 *
 * The shell is chrome-free: no border, no fill, the workbench reads as
 * one uninterrupted surface.
 */
export const AppShell = () => {
  const location = useLocation();
  const onWorkbench = location.pathname === '/' || location.pathname === '';
  const { conversation } = useConversation();
  const latestRunId = useMemo(() => {
    if (!conversation || conversation.runs.length === 0) return null;
    return conversation.runs[conversation.runs.length - 1].runId;
  }, [conversation]);
  const ledger = useContextLedger(latestRunId);

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.left}>
          <AppMenu />
        </div>
        <div className={styles.center}>
          {onWorkbench ? (
            <ChatHeader
              conversation={conversation}
              ledger={ledger.ledger}
              compactions={ledger.compactions.length}
              lastCompaction={
                ledger.compactions.length > 0
                  ? ledger.compactions[ledger.compactions.length - 1]
                  : null
              }
            />
          ) : (
            <PageTitle />
          )}
        </div>
        <div className={styles.right}>
          <div className={styles.statusPill} title="System status">
            <span className={styles.statusDot} />
            <span>Ready</span>
          </div>
        </div>
      </header>
      <main className={styles.main}>
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  );
};

/**
 * Compact title for the non-workbench pages. The wordmark dropdown
 * already provides navigation; the centre area only needs to name
 * where the user is.
 */
function PageTitle() {
  const location = useLocation();
  const titles: Record<string, string> = {
    '/tasks': 'Tasks',
    '/conversations': 'Conversations',
    '/knowledge': 'Knowledge',
    '/notebooks': 'Notebooks',
    '/approvals': 'Approvals',
    '/audit': 'Audit & Network',
    '/health': 'Health',
    '/model-health': 'Model Health',
    '/demo': 'SIH Demo',
    '/sih': 'SIH Dashboard',
    '/models': 'Models',
    '/system': 'System',
    '/settings': 'Settings',
  };
  const label = titles[location.pathname] ?? 'Arjun';
  return <span className={styles.pageTitle}>{label}</span>;
}
