import { createContext, useContext } from "react";
import type { ReactNode } from "react";

interface AuthContextType {
  user: { id: string; email: string; nickname: string };
  isAuthenticated: boolean;
  isGuest: boolean;
  isLoading: boolean;
  accessToken: string | null;
  showSetPassword: boolean;
  setShowSetPassword: (v: boolean) => void;
  login: () => void;
  logout: () => void;
  enterGuestMode: () => void;
  updateUser: () => void;
}

const _user = { id: "dev", email: "dev@localhost", nickname: "Local User" };

const defaultCtx: AuthContextType = {
  user: _user,
  isAuthenticated: true,
  isGuest: false,
  isLoading: false,
  accessToken: null,
  showSetPassword: false,
  setShowSetPassword: () => {},
  login: () => {},
  logout: () => {},
  enterGuestMode: () => {},
  updateUser: () => {},
};

const AuthContext = createContext<AuthContextType>(defaultCtx);

export function AuthProvider({ children }: { children: ReactNode }) {
  return (
    <AuthContext.Provider value={defaultCtx}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  return useContext(AuthContext);
}
