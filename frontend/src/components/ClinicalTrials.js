import React, { useState, useEffect } from 'react';

const ClinicalTrials = () => {
  const [batches, setBatches] = useState([]);
  const [formData, setFormData] = useState({
    drugName: '',
    expiry: '',
    sender: '',
    receiver: ''
  });

  useEffect(() => {
    fetchBatches();
  }, []);

  const fetchBatches = async () => {
    try {
      const response = await fetch('http://localhost:8000/trials');
      const data = await response.json();
      setBatches(data.batches || []);
    } catch (error) {
      console.error('Error fetching batches:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/trials', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      if (response.ok) {
        setFormData({ drugName: '', expiry: '', sender: '', receiver: '' });
        fetchBatches();
      }
    } catch (error) {
      console.error('Error creating batch:', error);
    }
  };

  const approveBatch = async (batchId) => {
    try {
      const response = await fetch(`http://localhost:8000/trials/${batchId}/approve`, {
        method: 'PUT',
      });
      if (response.ok) {
        fetchBatches();
      }
    } catch (error) {
      console.error('Error approving batch:', error);
    }
  };

  return (
    <div className="space-y-8">
      <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-6">Log New Drug Batch</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Drug Name</label>
              <input
                type="text"
                value={formData.drugName}
                onChange={(e) => setFormData({...formData, drugName: e.target.value})}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Expiry Date</label>
              <input
                type="date"
                value={formData.expiry}
                onChange={(e) => setFormData({...formData, expiry: e.target.value})}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Sender</label>
              <input
                type="text"
                value={formData.sender}
                onChange={(e) => setFormData({...formData, sender: e.target.value})}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Receiver</label>
              <input
                type="text"
                value={formData.receiver}
                onChange={(e) => setFormData({...formData, receiver: e.target.value})}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>
          </div>
          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors"
          >
            Create Batch
          </button>
        </form>
      </div>

      <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-6">Secure Ledger</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Batch ID</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Drug Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Expiry</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sender</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Receiver</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {batches.map((batch) => (
                <tr key={batch.batchID} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{batch.batchID}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{batch.drugName}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{batch.expiry}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{batch.sender}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{batch.receiver}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      batch.status === 'approved' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {batch.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {batch.status === 'pending' && (
                      <button
                        onClick={() => approveBatch(batch.batchID)}
                        className="bg-green-600 text-white px-3 py-1 rounded-md hover:bg-green-700 transition-colors"
                      >
                        Approve
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ClinicalTrials;
