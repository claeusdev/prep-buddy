import React, { useState, useEffect } from 'react';
import LandingPage from '@/components/LandingPage';
import CodingApp from '@/components/CodingApp';
import SystemDesignApp from '@/components/SystemDesignApp';
import LearningApp from '@/components/LearningApp';
import { BootSequence } from '@/components/BootSequence';
import Scanlines from '@/components/Scanlines';

type ViewState = 'landing' | 'coding' | 'system-design' | 'learning';

const AppContent: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewState>('landing');

  const handleNavigate = (view: ViewState) => {
    setCurrentView(view);
  };

  return (
    <>
      <Scanlines />
      {currentView === 'landing' && (
        <LandingPage onNavigate={handleNavigate} />
      )}
      {currentView === 'coding' && (
        <CodingApp onNavigateHome={() => setCurrentView('landing')} />
      )}
      {currentView === 'system-design' && (
        <SystemDesignApp onNavigateHome={() => setCurrentView('landing')} />
      )}
      {currentView === 'learning' && (
        <LearningApp onNavigateHome={() => setCurrentView('landing')} />
      )}
    </>
  );
};

const App: React.FC = () => {
  const [booting, setBooting] = useState(true);

  // Only boot on first load
  useEffect(() => {
    const hasBooted = sessionStorage.getItem('pb_booted');
    if (hasBooted) {
      setBooting(false);
    }
  }, []);

  const handleBootComplete = () => {
    setBooting(false);
    sessionStorage.setItem('pb_booted', 'true');
  };

  if (booting) {
    return <BootSequence onComplete={handleBootComplete} />;
  }

  return <AppContent />;
};

export default App;