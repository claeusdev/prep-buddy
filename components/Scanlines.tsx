
import React from 'react';

const Scanlines = () => {
  return (
    <div className="pointer-events-none fixed inset-0 z-[9999] overflow-hidden h-full w-full opacity-[0.15]">
      <div className="scanlines absolute inset-0"></div>
      <div className="flicker absolute inset-0 bg-white opacity-[0.02]"></div>
    </div>
  );
};

export default Scanlines;
