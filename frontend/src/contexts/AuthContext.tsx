'use client'

import React, { createContext, useContext, useState, useEffect } from 'react'
import { api } from '@/lib/api'

interface User {
  id: string
  username: string
  email: string
  role: string
}

interface AuthContextType {
  user: User | null
  token: string | null
  login: (username: string, password: string) => Promise<void>
  logout: () => void
  register: (username: string, email: string, password: string) => Promise<void>
  isAuthenticated: boolean
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const isAuthenticated = !!user && !!token

  useEffect(() => {
    // Check for stored token on app start
    const storedToken = localStorage.getItem('geoai_token')
    const storedUser = localStorage.getItem('geoai_user')
    
    if (storedToken && storedUser) {
      setToken(storedToken)
      setUser(JSON.parse(storedUser))
    }
    
    setIsLoading(false)
  }, [])

  const login = async (username: string, password: string) => {
    try {
      // Create form data for the FastAPI login endpoint
      const formData = new FormData()
      formData.append('username', username)
      formData.append('password', password)
      
      const response = await fetch('http://127.0.0.1:8002/api/v1/auth/login', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `Login failed: ${response.statusText}`)
      }
      
      const data = await response.json()
      const { access_token, token_type, user } = data
      
      setToken(access_token)
      setUser(user)
      
      localStorage.setItem('geoai_token', access_token)
      localStorage.setItem('geoai_user', JSON.stringify(user))
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  }

  const register = async (username: string, email: string, password: string) => {
    try {
      await api.post('/auth/register', { username, email, password })
      // Auto-login after registration
      await login(username, password)
    } catch (error) {
      throw new Error('Registration failed')
    }
  }

  const logout = () => {
    setUser(null)
    setToken(null)
    localStorage.removeItem('geoai_token')
    localStorage.removeItem('geoai_user')
  }

  return (
    <AuthContext.Provider value={{
      user,
      token,
      login,
      logout,
      register,
      isAuthenticated,
      isLoading
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}