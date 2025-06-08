import { useEffect, useState } from 'react';
import JDCard from './JDCard';
import ResumeUploadPopup from './ResumeUploadPopup';

import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
const JDViewer = () => {

  const searchParams = useSearchParams();
  const jdId = searchParams.get('jdId') ;

  const [showPopup, setShowPopup] = useState(false);
  const mockJD = {
    title: '',
    company: '',
    location: '',
    description: '',
    responsibilities: [] as string[],
    requirements: [] as string[]
  };

  type Job = {
    id: string;
    title: string;
    company: string;
    location: string;
  };

  const [jd, setJd] = useState(mockJD);

  const [jobs, setJobs] = useState<Job[]>([]);

  useEffect( () => {
    if(jdId) {
      fetch(process.env.API_HOST + '/jobs?jdId=' + jdId).then( (res) => {
        res.json().then( (resp) => {
          if(resp.jobs && resp.jobs.length > 0) {
            setJd(resp.jobs[0]);
          }
        });
      });
    }
    else {
      const fetchJobs = async () => {
        const res = await fetch(process.env.API_HOST + "/jobs");
        const data = await res.json();
        setJobs(data['jobs']);
      };
      fetchJobs();
    }
  }, [jdId]);

  if(jdId) {
    return (
      <div className="jd-viewer-container">
        <JDCard jd={jd} onApplyClick={() => setShowPopup(true)} />
        {showPopup && (
          <ResumeUploadPopup 
            jdId={jdId} 
            onClose={() => setShowPopup(false)} 
          />
        )}
      </div>
    );
  }
  else {
    return (
      <main className="max-w-5xl mx-auto p-6">
        <h1 className="text-3xl font-bold mb-6 text-center">Available Jobs</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {jobs.map((job) => (
            <Link key={job.id} href={`/?jdId=${job.id}`}>
              <div className="border rounded-xl p-5 shadow-sm hover:shadow-md transition bg-white cursor-pointer">
                <h2 className="text-xl font-semibold mb-1">{job.title}</h2>
                <p className="text-gray-700">{job.company}</p>
                <p className="text-sm text-gray-500">{job.location}</p>
              </div>
            </Link>
          ))}
        </div>
      </main>
    );
  }
};

export default JDViewer;
