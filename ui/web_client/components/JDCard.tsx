import React from 'react';

type JDProps = {
  jd: {
    title: string;
    company: string;
    location: string;
    description: string;
    responsibilities: string[];
    requirements: string[];
  };
  onApplyClick: () => void;
};

const JDCard: React.FC<JDProps> = ({ jd, onApplyClick }) => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 max-w-2xl w-full mx-auto space-y-4 border border-gray-200 hover:shadow-xl transition-shadow duration-300">
      <div className="space-y-1">
        <h2 className="text-2xl font-bold text-gray-900">{jd.title}</h2>
        <h3 className="text-lg text-gray-700">{jd.company}</h3>
        <p className="text-sm text-gray-500">
          📍 <span className="font-medium">{jd.location}</span>
        </p>
      </div>

      <p className="text-gray-700 text-sm leading-relaxed">{jd.description}</p>

      <div>
        <h4 className="font-semibold text-gray-800 mb-1">🛠 Responsibilities</h4>
        <ul className="list-disc list-inside text-gray-700 space-y-1 text-sm">
          {jd.responsibilities.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      </div>

      <div>
        <h4 className="font-semibold text-gray-800 mb-1">🎯 Requirements</h4>
        <ul className="list-disc list-inside text-gray-700 space-y-1 text-sm">
          {jd.requirements.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      </div>

      <div className="pt-4 text-right">
        <button
          onClick={onApplyClick}
          className="bg-indigo-600 text-white px-5 py-2 rounded-md text-sm font-semibold hover:bg-indigo-700 transition-colors"
        >
          Apply Now
        </button>
      </div>
    </div>
  );
};

export default JDCard;
