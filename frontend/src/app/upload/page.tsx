"use client"

import { useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'

export default function Upload() {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [processingOptions, setProcessingOptions] = useState({
    model_type: 'mask_rcnn',
    apply_regularization: true,
    apply_ml: true
  })
  const [results, setResults] = useState<any[]>([])
  const { user, token } = useAuth()
  const router = useRouter()

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.tiff', '.tif']
    },
    multiple: true
  })

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index))
  }

  const uploadFiles = async () => {
    if (files.length === 0) {
      toast.error('Please select files to upload')
      return
    }

    if (!token) {
      toast.error('Please login to upload files')
      router.push('/login')
      return
    }

    setUploading(true)
    const uploadResults = []

    try {
      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('apply_ml', processingOptions.apply_ml.toString())

        const response = await fetch('http://127.0.0.1:8002/api/v1/ml-processing/upload-image', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData
        })

        if (!response.ok) {
          throw new Error(`Failed to upload ${file.name}`)
        }

        const result = await response.json()
        uploadResults.push({
          filename: file.name,
          task_id: result.task_id,
          status: 'processing'
        })

        toast.success(`Started processing ${file.name}`)
      }

      setResults(uploadResults)
      setFiles([])
    } catch (error: any) {
      toast.error(error.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const checkTaskStatus = async (taskId: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:8002/api/v1/ml-processing/task-status/${taskId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (!response.ok) {
        throw new Error('Failed to check task status')
      }

      const status = await response.json()
      return status
    } catch (error) {
      toast.error('Failed to check task status')
      return null
    }
  }

  const processState = async (stateName: string) => {
    if (!token) {
      toast.error('Please login first')
      return
    }

    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('state_name', stateName)

      const response = await fetch('http://127.0.0.1:8002/api/v1/ml-processing/process-state', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Failed to process state: ${stateName}`)
      }

      const result = await response.json()
      toast.success(`Started processing ${stateName}`)
      
      setResults(prev => [...prev, {
        filename: `${stateName}_state_processing`,
        task_id: result.task_id,
        status: 'processing'
      }])
    } catch (error: any) {
      toast.error(error.message || 'State processing failed')
    } finally {
      setUploading(false)
    }
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Please Login</h2>
          <button
            onClick={() => router.push('/login')}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go to Login
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="neural-network-bg opacity-10"></div>
      </div>
      
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Upload & Process</h1>
          <p className="text-gray-300">Upload satellite images for building footprint analysis</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="glass-morphism p-6 rounded-xl border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-4">Image Upload</h2>
            
            {/* Drag & Drop Zone */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
                isDragActive 
                  ? 'border-blue-500 bg-blue-500/10' 
                  : 'border-gray-600 hover:border-gray-500'
              }`}
            >
              <input {...getInputProps()} />
              <div className="text-6xl mb-4">📸</div>
              {isDragActive ? (
                <p className="text-blue-400">Drop the files here...</p>
              ) : (
                <div>
                  <p className="text-gray-300 mb-2">Drag & drop images here, or click to select</p>
                  <p className="text-sm text-gray-500">Supports: JPEG, PNG, TIFF</p>
                </div>
              )}
            </div>

            {/* Processing Options */}
            <div className="mt-6 space-y-4">
              <h3 className="text-lg font-semibold text-white">Processing Options</h3>
              
              <div className="space-y-3">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={processingOptions.apply_ml}
                    onChange={(e) => setProcessingOptions(prev => ({
                      ...prev,
                      apply_ml: e.target.checked
                    }))}
                    className="mr-2"
                  />
                  <span className="text-gray-300">Apply ML Processing</span>
                </label>
                
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={processingOptions.apply_regularization}
                    onChange={(e) => setProcessingOptions(prev => ({
                      ...prev,
                      apply_regularization: e.target.checked
                    }))}
                    className="mr-2"
                  />
                  <span className="text-gray-300">Apply Geometric Regularization</span>
                </label>
                
                <div>
                  <label className="block text-gray-300 mb-1">Model Type</label>
                  <select
                    value={processingOptions.model_type}
                    onChange={(e) => setProcessingOptions(prev => ({
                      ...prev,
                      model_type: e.target.value
                    }))}
                    className="w-full p-2 bg-gray-800 text-white rounded border border-gray-600"
                  >
                    <option value="mask_rcnn">Mask R-CNN</option>
                    <option value="adaptive_fusion">Adaptive Fusion</option>
                    <option value="hybrid">Hybrid Model</option>
                  </select>
                </div>
              </div>
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-white mb-3">Selected Files</h3>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {files.map((file, index) => (
                    <div key={index} className="flex items-center justify-between bg-gray-800/50 p-2 rounded">
                      <span className="text-gray-300 text-sm">{file.name}</span>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-red-400 hover:text-red-300 text-sm"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Upload Button */}
            <button
              onClick={uploadFiles}
              disabled={uploading || files.length === 0}
              className="w-full mt-6 py-3 px-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:from-blue-700 hover:to-purple-700 transition-all"
            >
              {uploading ? 'Processing...' : `Upload ${files.length} Files`}
            </button>
          </div>

          {/* State Processing & Results */}
          <div className="space-y-6">
            {/* Quick State Processing */}
            <div className="glass-morphism p-6 rounded-xl border border-white/10">
              <h2 className="text-2xl font-bold text-white mb-4">State Processing</h2>
              <p className="text-gray-300 mb-4">Process predefined state regions</p>
              
              <div className="grid grid-cols-2 gap-3">
                {['California', 'Texas', 'Florida', 'New York'].map(state => (
                  <button
                    key={state}
                    onClick={() => processState(state)}
                    disabled={uploading}
                    className="py-2 px-4 bg-gray-800 text-white rounded hover:bg-gray-700 disabled:opacity-50 transition-colors"
                  >
                    {state}
                  </button>
                ))}
              </div>
            </div>

            {/* Results */}
            {results.length > 0 && (
              <div className="glass-morphism p-6 rounded-xl border border-white/10">
                <h2 className="text-2xl font-bold text-white mb-4">Processing Results</h2>
                
                <div className="space-y-3">
                  {results.map((result, index) => (
                    <div key={index} className="bg-gray-800/50 p-3 rounded">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-300">{result.filename}</span>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs rounded ${
                            result.status === 'completed' 
                              ? 'bg-green-600 text-white'
                              : result.status === 'processing'
                              ? 'bg-blue-600 text-white'
                              : 'bg-red-600 text-white'
                          }`}>
                            {result.status}
                          </span>
                          <button
                            onClick={() => checkTaskStatus(result.task_id)}
                            className="text-blue-400 hover:text-blue-300 text-sm"
                          >
                            Check Status
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}