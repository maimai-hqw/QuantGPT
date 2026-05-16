import { Star, MessageSquare } from "lucide-react";
import type { Session, Task } from "../types/backtest";
import SessionSidebar from "./SessionSidebar";
import FactorLibrary from "./FactorLibrary";

interface Props {
  sidebarTab: "sessions" | "factors";
  onSidebarTabChange: (tab: "sessions" | "factors") => void;
  sessions: Session[];
  activeSessionId: string | null;
  tasks: Task[];
  activeTaskId?: string;
  onCreateSession: () => void;
  onSwitchSession: (id: string) => void;
  onRenameSession: (id: string, name: string) => void;
  onDeleteSession: (id: string) => void;
  onSelectTask: (task: Task) => void;
  factorLibKey: number;
}

export default function AppSidebar({
  sidebarTab,
  onSidebarTabChange,
  sessions,
  activeSessionId,
  tasks,
  activeTaskId,
  onCreateSession,
  onSwitchSession,
  onRenameSession,
  onDeleteSession,
  onSelectTask,
  factorLibKey,
}: Props) {
  return (
    <aside className="w-72 shrink-0 hidden lg:block">
      <div className="sticky top-6 max-h-[calc(100vh-3rem)] flex flex-col">
        <div className="flex gap-1 mb-3 shrink-0">
          <button
            onClick={() => onSidebarTabChange("sessions")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              sidebarTab === "sessions" ? "bg-blue-50 text-blue-700" : "text-gray-500 hover:bg-gray-100"
            }`}
          >
            <MessageSquare className="h-3.5 w-3.5" />
            会话
          </button>
          <button
            onClick={() => onSidebarTabChange("factors")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              sidebarTab === "factors" ? "bg-amber-50 text-amber-700" : "text-gray-500 hover:bg-gray-100"
            }`}
          >
            <Star className="h-3.5 w-3.5" />
            因子库
          </button>
        </div>
        <div className="overflow-y-auto min-h-0">
          {sidebarTab === "sessions" ? (
            <SessionSidebar
              sessions={sessions}
              activeSessionId={activeSessionId}
              tasks={tasks}
              activeTaskId={activeTaskId}
              onCreateSession={onCreateSession}
              onSwitchSession={onSwitchSession}
              onRenameSession={onRenameSession}
              onDeleteSession={onDeleteSession}
              onSelectTask={onSelectTask}
            />
          ) : (
            <FactorLibrary key={factorLibKey} />
          )}
        </div>
      </div>
    </aside>
  );
}
