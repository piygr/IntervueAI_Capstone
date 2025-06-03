import { useState } from 'react';
import JDCard from './JDCard';
import ResumeUploadPopup from './ResumeUploadPopup';

const JDViewer = () => {
  const [showPopup, setShowPopup] = useState(false);

  const mockJD = {
    title: 'Frontend Developer (React)',
    company: 'TechNova Labs',
    location: 'Remote - India',
    description: `We are looking for a skilled React developer with experience in building modern web applications.`,
    responsibilities: [
      'Develop reusable components in React',
      'Collaborate with backend teams on API integration',
      'Write clean, testable code with unit tests'
    ],
    requirements: [
      '2+ years of experience with React',
      'Familiarity with Git, RESTful APIs',
      'Understanding of browser rendering and performance'
    ]
  };

  return (
    <div className="jd-viewer-container">
      <JDCard jd={mockJD} onApplyClick={() => setShowPopup(true)} />
      {showPopup && (
        <ResumeUploadPopup 
          jdId={"JD-001"} 
          onClose={() => setShowPopup(false)} 
        />
      )}
    </div>
  );
};

export default JDViewer;
