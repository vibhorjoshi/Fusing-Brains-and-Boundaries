'use client';

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function Home() {
  const router = useRouter();
  
  useEffect(() => {
    // Redirect to dashboard automatically
    router.push('/dashboard');
  }, [router]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500 mx-auto"></div>
        <h1 className="text-2xl font-bold text-white mt-6">Loading GeoAI Dashboard...</h1>
        <p className="text-gray-400 mt-2">Redirecting to 3D Building Detection Platform</p>
      </div>
    </div>
  );
}