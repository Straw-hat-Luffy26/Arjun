import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ChevronDown, BookOpen, NotebookPen, Boxes, ShieldCheck, Activity, HeartPulse, Cpu, Settings, UserRound, LogOut, MessageSquare, ListTodo, Plus,
} from 'lucide-react';
import {
  governanceService,
  headlineRole,
  isActiveRole,
  type Session,
} from '../../services/governance.service';
import styles from './AppMenu.module.css';

interface MenuItem {
  label: string;
  icon: React.ReactNode;
  path: string;
  shortcut: string;
  /** When omitted, the item is visible to both roles. */
  requires?: 'administrator';
}

interface Section {
  label: string;
  icon: React.ReactNode;
  items: MenuItem[];
}

/**
 * Sections in the ARJUN dropdown. The workbench lives in the trigger
 * itself (a `+ New conversation` button) — the dropdown is for
 * navigation and account.
 *
 * The full conversation history lives at `/conversations`, reachable
 * from the Workbench section below; it is intentionally not duplicated
 * here, so the dropdown stays a navigation surface and not a history
 * surface. Items with `requires: 'administrator'` are hidden from
 * Employees.
 */
const SECTIONS: Section[] = [
  {
    label: 'Workbench',
    icon: <ListTodo size={11} />,
    items: [
      { label: 'New conversation', icon: <Plus size={15} />,         path: '/',              shortcut: 'N' },
      { label: 'Tasks',            icon: <ListTodo size={15} />,      path: '/tasks',         shortcut: 'H' },
      { label: 'Conversations',    icon: <MessageSquare size={15} />, path: '/conversations', shortcut: 'Y' },
      { label: 'Knowledge',        icon: <BookOpen size={15} />,      path: '/knowledge',     shortcut: 'K' },
      { label: 'Notebooks',        icon: <NotebookPen size={15} />,   path: '/notebooks',     shortcut: 'G' },
    ],
  },
  {
    label: 'Administration',
    icon: <Settings size={11} />,
    items: [
      { label: 'Models',          icon: <Boxes size={15} />,       path: '/models',       shortcut: 'M', requires: 'administrator' },
      { label: 'Approvals',       icon: <ShieldCheck size={15} />, path: '/approvals',    shortcut: 'R', requires: 'administrator' },
      { label: 'Audit & Network', icon: <Activity size={15} />,     path: '/audit',        shortcut: 'L', requires: 'administrator' },
      { label: 'Health',          icon: <HeartPulse size={15} />,  path: '/health',       shortcut: 'J', requires: 'administrator' },
      { label: 'Model Health',    icon: <Cpu size={15} />,         path: '/model-health', shortcut: 'O', requires: 'administrator' },
      { label: 'Settings',        icon: <Settings size={15} />,    path: '/settings',     shortcut: 'A', requires: 'administrator' },
    ],
  },
];

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);
const MOD_LABEL = IS_MAC ? '\u2318' : 'Ctrl';

export const AppMenu = () => {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [session, setSession] = useState<Session | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void (async () => {
      try {
        setSession(await governanceService.currentSession());
      } catch {
        /* noop */
      }
    })();
  }, []);

  // The single "what to call this account" role for the menu. The
  // back-end hands us a role list; the user sees one of two
  // possibilities: Administrator or Employee.
  const role = useMemo<'administrator' | 'employee' | null>(() => {
    if (!session) return null;
    const active = session.user.roles.filter(isActiveRole);
    if (active.length === 0) return 'employee';
    return headlineRole(active);
  }, [session]);

  const signOut = async () => {
    try {
      await governanceService.signOut();
    } finally {
      window.location.reload();
    }
  };

  const go = useCallback((path: string) => {
    setOpen(false);
    navigate(path);
  }, [navigate]);

  // Close on Escape, and on a click landing outside the menu.
  useEffect(() => {
    if (!open) return;

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    const onPointer = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };

    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onPointer);
    };
  }, [open]);

  // Global shortcuts. Only fire for items the current role can see;
  // the route's own guard handles the rest.
  useEffect(() => {
    const isAdmin = role === 'administrator';
    const allItems = SECTIONS.flatMap(s => s.items).filter(
      (i) => isAdmin || !i.requires,
    );
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return;
      const hit = allItems.find(i => i.shortcut.toLowerCase() === e.key.toLowerCase());
      if (!hit) return;
      e.preventDefault();
      go(hit.path);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [go, role]);

  // `Ctrl+N` is reserved by the browser for "new window" on some
  // platforms; the menu's "New conversation" is `Ctrl+Shift+N`.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey) return;
      if (e.shiftKey && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        go('/');
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [go]);

  // Per-role visibility. The default `employee` view hides every
  // section whose label is "Administration". Both roles see "Workbench".
  const visibleSections = useMemo(() => {
    if (role === 'administrator') return SECTIONS;
    return SECTIONS.filter(s => s.label !== 'Administration');
  }, [role]);

  return (
    <div className={styles.root} ref={rootRef}>
      <button
        className={styles.trigger}
        onClick={() => setOpen(o => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        title="Menu"
      >
        <span className={styles.wordmark}>ARJUN</span>
        <ChevronDown size={13} className={open ? styles.chevronOpen : styles.chevron} />
      </button>

      {open && (
        <div className={styles.panel} role="menu">
          {visibleSections.map(section => (
            <div key={section.label} className={styles.sectionGroup}>
              <div className={styles.sectionHeader}>
                {section.icon}
                <span>{section.label}</span>
              </div>
              {section.items.map(item => (
                <button
                  key={item.path + item.label}
                  className={styles.item}
                  role="menuitem"
                  onClick={() => go(item.path)}
                >
                  <span className={styles.itemIcon}>{item.icon}</span>
                  <span className={styles.itemLabel}>{item.label}</span>
                  <span className={styles.itemShortcut}>{MOD_LABEL} {item.shortcut}</span>
                </button>
              ))}
            </div>
          ))}

          {/* Account at the bottom. */}
          <div className={styles.divider} />
          <div className={styles.account}>
            <span className={styles.accountIcon}><UserRound size={14} /></span>
            <span className={styles.accountText}>
              <span className={styles.accountName}>
                {session ? session.user.displayName : 'Not signed in'}
              </span>
              <span className={styles.accountRoles}>
                {session && role
                  ? role === 'administrator'
                    ? 'Administrator'
                    : 'Employee'
                  : 'No permissions are held'}
              </span>
            </span>
            {session && (
              <button
                className={styles.accountSwitch}
                onClick={() => void signOut()}
                aria-label="Sign out"
                title="Sign out"
              >
                <LogOut size={13} />
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
