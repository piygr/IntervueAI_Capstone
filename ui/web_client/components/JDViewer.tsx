import { useEffect, useState } from 'react';
import JDCard from './JDCard';
import ResumeUploadPopup from './ResumeUploadPopup';

import { useSearchParams } from 'next/navigation';
const JDViewer = () => {

  const searchParams = useSearchParams();
  const jdId = searchParams.get('jdId') || 'JD-001';

  const [showPopup, setShowPopup] = useState(false);
  const mockJD = {
    title: '',
    company: '',
    location: '',
    description: '',
    responsibilities: [] as string[],
    requirements: [] as string[]
  };

  const [jd, setJd] = useState(mockJD);

  useEffect( () => {
    if(jdId) {
      //fetch
      fetch(process.env.API_HOST + '/jobs?jdId=' + jdId).then( (res) => {
        res.json().then( (resp) => {
          if(resp.jobs && resp.jobs.length > 0) {
            setJd(resp.jobs[0]);
          }
        });
      });
    }
    else {
      //let job = 
      //setJd(job);
    }
  }, [jdId]);

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
};

export default JDViewer;
