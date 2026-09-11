import React, { useCallback, useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppStateProvider } from './contexts/AppStateContext';
import { ConfigProvider } from './contexts/ConfigContext';
import { ConversationProvider } from './contexts/ConversationContext';
import { ActiveRunProvider } from './contexts/ActiveRunContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ToastProvider } from './contexts/ToastContext';
import { PermissionsProvider, usePermissions } from './contexts/PermissionsContext';
import { AppShell } from './components/layout';
import { RequirePermission } from './components/auth/RequirePermission';
import { Workbench } from './pages/Workbench';
import { DocumentScan } from './pages/DocumentScan';
import { Tasks } from './pages/Tasks';
import { Conversations } from './pages/Conversations';
import { Knowledge } from './pages/Knowledge';
import { Notebooks } from './pages/Notebooks';
import { AuditNetwork } from './pages/AuditNetwork';
import { SignIn } from './pages/SignIn';
import { governanceService, type Session } from './services/governance.service';
import { Settings } from './pages/Settings';
import { SystemInfo } from './pages/SystemInfo';
import { Storage } from './pages/Storage';
import { Health } from './pages/Health';
import { ModelHealth } from './pages/ModelHealth';
import { Approvals } from './pages/Approvals';
import { Demo } from './pages/Demo';
import { SIHDashboard } from './pages/SIHDashboard';

/**
 * Everything below the shell assumes a signed-in user: the policy gateway
 * decides from their roles, and the audit log records their id. So sign-in is a
 * gate around the whole application rather than a page inside it — there is no
 * meaningful state to show somebody ARJUN cannot yet name.
 */
function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [checked, setChecked] = useState(false);

  const restore = useCallback(async () => {
    try {
      setSession(await governanceService.currentSession());
    } catch {
      // Backend unavailable (a browser-only dev run). Falls through to the
      // sign-in screen, which will surface the real error when it tries.
    } finally {
      setChecked(true);
    }
  }, []);

  useEffect(() => {
    void restore();
  }, [restore]);

  // Nothing is drawn until we know, so the shell never flashes before the gate.
  if (!checked) return null;

  if (!session) {
    return (
      <ThemeProvider>
        <ToastProvider>
          {/* SignIn hands the new session back via `onSignedIn`. The
            * PermissionsProvider below is mounted on the next render,
            * and its own mount-time `refresh` reads the freshly-stored
            * session from the back-end. */}
          <SignIn onSignedIn={setSession} />
        </ToastProvider>
      </ThemeProvider>
    );
  }

  return (
    <AppStateProvider>
      <ConfigProvider>
        <ThemeProvider>
          <ToastProvider>
            <PermissionsProvider>
              <ConversationProvider>
                <ActiveRunProvider>
                <BrowserRouter>
                <Routes>
                  <Route path="/" element={<AppShell />}>
                    <Route index element={<Workbench />} />

                    <Route path="tasks" element={<Tasks />} />
                    <Route path="conversations" element={<Conversations />} />

                    <Route path="knowledge" element={<Knowledge />} />
                    <Route path="notebooks" element={<Notebooks />} />
                    {/* Diagnostic surface, deliberately not in the menu.
                      * Ordinary OCR happens in chat: attach a document to a
                      * message and the runtime reads it before answering.
                      * This page exists to exercise one page at a chosen
                      * accuracy stop and to see the model's own bounding
                      * boxes — which is how the coordinate space was
                      * calibrated, and how a bad page is diagnosed. Reachable
                      * at /scan?doc=<sha256>&page=<n> for that purpose. */}
                    <Route path="scan" element={<DocumentScan />} />
                    {/* Approvals queue: gated on `ApproveOutput`, which in the
                      * 2-role model only `Administrator` holds. (An Employee can
                      * decide approvals for tasks they themselves own, but cannot
                      * see the cross-account queue.) */}
                    <Route
                      path="approvals"
                      element={
                        <RequirePermission permission="approveOutput">
                          <Approvals />
                        </RequirePermission>
                      }
                    />
                    {/* Audit + sovereignty read surface: `ViewAuditLog`. */}
                    <Route
                      path="audit"
                      element={
                        <RequirePermission permission="viewAuditLog">
                          <AuditNetwork />
                        </RequirePermission>
                      }
                    />
                    <Route path="health" element={<Health />} />
                    {/* Model Health is model-management metadata, so `ImportModel`. */}
                    <Route
                      path="model-health"
                      element={
                        <RequirePermission permission="importModel">
                          <ModelHealth />
                        </RequirePermission>
                      }
                    />
                    <Route path="demo" element={<Demo />} />
                    <Route path="sih" element={<SIHDashboard />} />

                    {/* Carried over from Sarathi. Model-management surfaces:
                      * installing, deleting, and loading a model is `ImportModel`,
                      * which in the 2-role model only `Administrator` holds, so
                      * the catalogue and the storage page are gated on it. */}
                    <Route
                      path="models"
                      element={
                        <RequirePermission permission="importModel">
                          <Storage />
                        </RequirePermission>
                      }
                    />
                    <Route path="system" element={<SystemInfo />} />
                    <Route path="settings" element={<Settings />} />
                  </Route>
                </Routes>
              </BrowserRouter>
                </ActiveRunProvider>
            </ConversationProvider>
          </PermissionsProvider>
          </ToastProvider>
        </ThemeProvider>
      </ConfigProvider>
    </AppStateProvider>
  );
}

export default App;
